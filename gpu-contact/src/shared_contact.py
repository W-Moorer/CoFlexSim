"""Finite-overlap pressure and guards with matrix-free all-row feasibility."""
import time

import numpy as np
import scipy.sparse as sp

from bounded_patch_contact import BoundedPatchContact
from consistent_mortar_contact import ContactGeometryError, coflex_mortar_native


class FacePair:
    def __init__(self,geometry,slave,master,fs,fm,owners,mids,s,m,derivatives,gaps_only):
        self.native=coflex_mortar_native.SharedFaceBatch(s.reshape(-1,9),m.reshape(-1,9),
                                                        geometry.radius,geometry.patch.radius,derivatives,
                                                        geometry.retain_negative_pressure_support)
        self.stats=geometry.statistics
        self.stats["common_overlap_tests"]+=self.native.common_prepares
        self.stats["reused_patch_overlaps"]+=self.native.near_tests
        self.slave,self.master=slave,master
        self.guard=self.native.values(False);self.weak=self.native.values(True)
        selected=self.guard["input"]
        self.owners=owners[selected];self.mids=mids[selected]
        self.nodes=np.column_stack([fs[self.owners]+geometry.offsets[slave],fm[self.mids]+geometry.offsets[master]])
        self.cols=(3*self.nodes[:,:,None]+np.arange(3)).reshape(-1,18)
        for data in (self.guard,self.weak):
            used=data["samples"]>0
            covered=np.bincount(self.owners[used],data["area"][used].sum(axis=1),minlength=len(fs))
            if np.any(covered[self.owners[used]]>data["info"][used,5]*(1.+1e-7)):
                raise ContactGeometryError("Multiple primary sheets in the secondary support")
        self.pair,self.slot=np.nonzero(np.arange(12)<self.guard["vertex_counts"][:,None])
        self.vertex_gaps=self.guard["vertex_gaps"][self.pair,self.slot]
        if self.vertex_gaps.min(initial=0.) < -geometry.radius:
            raise ContactGeometryError("Physical overlap exceeds negative search boundary")
        self.guard_gaps=self.vertex_gaps+geometry.guard_offset
        self.wi=np.flatnonzero(self.weak["samples"])
        self.raw_nodes,inv=np.unique(fs[self.owners[self.wi]].ravel(),return_inverse=True)
        self.inv=inv.reshape(-1,3)
        aa=self.weak["area"][self.wi]
        self.area=np.bincount(inv,aa.ravel(),minlength=len(self.raw_nodes))
        self.keep=np.flatnonzero(self.area>1e-24)
        self.weak_gaps=np.bincount(inv,self.weak["weighted_gap"][self.wi].ravel(),minlength=len(self.raw_nodes))/np.maximum(self.area,1e-300)
        self.partials={}
        if gaps_only:return
        norm=np.maximum(self.area,1e-300)
        pts=self.weak["points"][self.wi].reshape(-1,3)
        self.points=np.column_stack([np.bincount(inv,pts[:,k],minlength=len(norm))/norm for k in range(3)])
        info=self.weak["info"][self.wi]
        self.normals=np.column_stack([np.bincount(inv,(aa*info[:,k,None]).ravel(),minlength=len(norm))/norm for k in range(3)])
        self.min_gap=np.full(len(norm),np.inf);self.max_gap=np.full(len(norm),-np.inf)
        np.minimum.at(self.min_gap,inv,np.repeat(info[:,3],3));np.maximum.at(self.max_gap,inv,np.repeat(info[:,4],3))
        self.faces=sp.csr_matrix((aa.ravel(),(inv,np.repeat(self.mids[self.wi],3))),shape=(len(norm),len(fm)))
        self.offset=geometry.offsets[slave]
        self.guard_area=self.guard["area"].sum(axis=1)
        self.guard_points=self.guard["vertex_points"].reshape(-1,12,3)
        self.cache={}

    def record(self,kind,i):
        if (kind,i) in self.cache:return self.cache[kind,i]
        if kind=="weak":
            k=self.keep[i];a,b=self.faces.indptr[k:k+2]
            primary=self.faces.indices[a+np.argmax(self.faces.data[a:b])]
            touched=self.nodes[self.wi[(self.inv==k).any(axis=1)]]
            r=dict(key=(self.slave,self.master,int(self.raw_nodes[k])),ids=np.r_[self.offset+self.raw_nodes[k],np.unique(touched)],
                   normal=self.normals[k],master_face=int(primary),point=self.points[k],support_area=self.area[k],
                   min_sample_gap=self.min_gap[k],max_sample_gap=self.max_gap[k],derivative="total_piecewise_smooth")
        else:
            p=self.pair[i];j=self.slot[i]
            r=dict(key=(self.slave,self.master,int(self.owners[p]),int(self.mids[p]),int(j)),ids=self.nodes[p],
                   normal=self.guard["info"][p,:3],master_face=int(self.mids[p]),point=self.guard_points[p,j],
                   support_area=self.guard_area[p],min_sample_gap=self.vertex_gaps[i],max_sample_gap=self.guard["info"][p,4],
                   derivative="total_physical_overlap_vertices")
        self.cache[kind,i]=r
        return r

    def triplets(self,kind,ids):
        if kind=="guard":
            values=self.native.vertex_rows(self.pair[ids],self.slot[ids])
            return np.repeat(np.arange(len(ids)),18),self.cols[self.pair[ids]].ravel(),values.ravel()
        raw=self.keep[ids]
        mask=np.zeros(len(self.area),bool);mask[raw]=True
        j,k=np.nonzero(mask[self.inv])
        face=self.wi[j]
        missing=np.array(sorted(set(face.tolist())-self.partials.keys()),dtype=int)
        if len(missing):
            self.partials.update(zip(missing,self.native.pressure_partials(missing).reshape(-1,6,18)))
        partial=np.asarray([self.partials[p] for p in face])
        nodes=self.inv[j,k]
        v=(partial[np.arange(len(j)),3+k]-self.weak_gaps[nodes,None]*partial[np.arange(len(j)),k])/self.area[nodes,None]
        lut=np.empty(len(self.area),int);lut[raw]=np.arange(len(raw))
        return np.repeat(lut[nodes],18),self.cols[face].ravel(),v.ravel()

    def directional(self,dx):
        step=dx[self.cols]
        unknown=np.asarray([i for i in self.wi if i not in self.partials],int)
        vertex,values=self.native.directional(step[:,:9],step[:,9:],unknown)
        v=np.empty((len(self.wi),6))
        if len(unknown):v[np.searchsorted(self.wi,unknown)]=values
        for j,i in enumerate(self.wi):
            if i in self.partials:v[j]=self.partials[i]@step[i]
        nodes=self.inv.ravel()
        change=v[:,3:]-self.weak_gaps[self.inv]*v[:,:3]
        weak=np.bincount(nodes,change.ravel(),minlength=len(self.area))[self.keep]/self.area[self.keep]
        return weak,vertex


class Adjoint:
    def __init__(self,operator):self.operator=operator
    def __matmul__(self,weights):return self.operator.adjoint(weights)


class AllRows:
    no_open_rows=True
    def __init__(self,faces,size,stats):
        self.faces=faces;self.stats=stats;self.blocks=[];self.ends=[];end=0
        for kind in ("weak","guard"):
            for f in faces:
                n=len(f.keep) if kind=="weak" else len(f.pair)
                if n:
                    self.blocks.append((end,end+n,kind,f));end+=n;self.ends.append(end)
        self.shape=(end,size)
        self.world_cache={}

    @property
    def T(self):return Adjoint(self)

    def endpoint_columns(self):
        fields={k:[] for k in ("support","node","master","point","minimum","maximum","kind")}
        for lo,hi,kind,f in self.blocks:
            if kind=="weak":
                ids=f.keep
                masters=[]
                for i in ids:
                    a,b=f.faces.indptr[i:i+2]
                    masters.append(f.faces.indices[a+np.argmax(f.faces.data[a:b])])
                values=(f.area[ids],f.offset+f.raw_nodes[ids],masters,f.points[ids],f.min_gap[ids],f.max_gap[ids])
                name="total_piecewise_smooth"
            else:
                ids=f.pair
                values=(f.guard_area[ids],f.nodes[ids,0],f.mids[ids],f.guard_points[ids,f.slot],f.vertex_gaps,f.guard["info"][ids,4])
                name="total_physical_overlap_vertices"
            for key,value in zip(fields,values):fields[key].append(value)
            fields["kind"].extend([name]*(hi-lo))
        for key in ("support","node","master","point","minimum","maximum"):
            fields[key]=np.concatenate(fields[key]) if fields[key] else np.empty((0,3) if key=="point" else 0)
        return fields

    def __len__(self):return self.shape[0]

    def __getitem__(self,i):
        if i<0 or i>=len(self):raise IndexError(i)
        lo,_,kind,f=self.blocks[np.searchsorted(self.ends,i,side="right")]
        return f.record(kind,i-lo)

    def matching_indices(self,keys):
        for lo,hi,kind,f in self.blocks:
            for key in keys:
                if key[:2]!=(f.slave,f.master):continue
                if kind=="weak" and len(key)==3:
                    where=np.flatnonzero(f.raw_nodes[f.keep]==key[2])
                elif kind=="guard" and len(key)==5:
                    where=np.flatnonzero((f.owners[f.pair]==key[2])&(f.mids[f.pair]==key[3])&(f.slot==key[4]))
                else:continue
                for i in where:yield int(lo+i)

    def rows(self,ids):
        ids=np.asarray(ids,int)
        missing=np.array(sorted(set(ids.tolist())-self.world_cache.keys()),int)
        if len(missing):
            tick=time.perf_counter()
            rr=[];cc=[];vv=[]
            for lo,hi,kind,f in self.blocks:
                mask=np.flatnonzero((missing>=lo)&(missing<hi))
                if len(mask):
                    r,c,v=f.triplets(kind,missing[mask]-lo);rr.append(mask[r]);cc.append(c);vv.append(v)
            mat=sp.csr_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(len(missing),self.shape[1]))
            for j,i in enumerate(missing):self.world_cache[i]=mat[j]
            self.stats["working_rows_assembled"]+=len(missing)
            self.stats["working_assembly_s"]+=time.perf_counter()-tick
        if not len(ids):return sp.csr_matrix((0,self.shape[1]))
        return sp.vstack([self.world_cache[i] for i in ids],format="csr")

    def adjoint(self,weights):
        weights=np.asarray(weights).ravel();ids=np.flatnonzero(weights)
        if not len(ids):return np.zeros(self.shape[1])
        return np.asarray(self.rows(ids).T@weights[ids]).ravel()

    def __matmul__(self,dx):
        tick=time.perf_counter()
        results=[f.directional(np.asarray(dx).ravel()) for f in self.faces]
        self.stats["directional_checks"]+=1
        self.stats["directional_s"]+=time.perf_counter()-tick
        return np.concatenate([r[k] for k in (0,1) for r in results]) if results else np.empty(0)

    def mapped(self,J):
        return MappedRows(self,J)


class MappedRows:
    def __init__(self,world,J):
        self.world=world;self.J=J
        self.shape=(len(world),J.shape[1])
    @property
    def T(self):return Adjoint(self)
    def __matmul__(self,d):return self.world@(self.J@d)
    def adjoint(self,w):
        f=self.world.T@w
        return self.J.transpose_action(f) if hasattr(self.J,"transpose_action") else self.J.T@f
    def rows(self,ids):
        rows=self.world.rows(ids)
        if hasattr(self.J,"restrict_nodes"):
            from quasistatic_pd import ContactRows
            return ContactRows(rows,self.J).small_dense()
        return rows@self.J


class SharedBoundedContact(BoundedPatchContact):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.statistics.update(common_overlap_tests=0,reused_patch_overlaps=0,working_rows_assembled=0,rows_considered=0,
                               directional_checks=0,directional_s=0.,working_assembly_s=0.)

    def build(self,x,J,derivatives=True,gaps_only=False):
        tick=time.perf_counter();self.statistics["calls"]+=1
        try:
            faces=[FacePair(self,*args,derivatives,gaps_only) for args in self.pair_inputs(x)]
            gaps=np.concatenate([a for kind in (0,1) for f in faces for a in
                                 [f.weak_gaps[f.keep] if kind==0 else f.guard_gaps]]) if faces else np.empty(0)
            if gaps_only:return gaps
            world=AllRows(faces,x.size,self.statistics)
            if derivatives:self.statistics["rows_considered"]+=len(gaps)
            return gaps,world,(world.mapped(J) if derivatives and J.shape[1] else sp.csr_matrix((len(gaps),J.shape[1]))),world
        finally:self.statistics["total_build_seconds"]+=time.perf_counter()-tick


def solve_working_qp(sim,system,G,gaps,free_direction,free_gap,tol,history):
    """Generate full rows on the working set. Certify the direction on every row."""
    from quasistatic_pd import solve_contact_lcp
    multipliers=np.zeros(len(gaps))
    if free_gap.min(initial=0.)>=0.:
        return free_direction,multipliers,dict(iterations=0,active_count=0)
    seeds=sim.multiplier_history.match(*history,free_gap) if history is not None else {}
    ids=list(seeds)
    worst=int(np.argmin(free_gap))
    if worst not in ids:ids.append(worst)
    weights=np.array([seeds.get(i,0.) for i in ids])
    response=None
    for outer in range(len(gaps)):
        if 8*(3*G.shape[1]*len(ids)+6*len(ids)**2)>sim.policy["contact_workspace_bytes"]:
            raise MemoryError("Working-set response exceeds the contact workspace")
        added=ids if response is None else ids[-1:]
        rows=G.rows(added)
        rhs=rows.T.toarray() if sp.issparse(rows) else rows.T
        columns=system.solve(rhs)
        if response is None:
            response=columns;active=rows
        else:
            response=np.column_stack((response,columns))
            active=sp.vstack((active,rows),format="csr") if sp.issparse(rows) else np.vstack((active,rows))
        small=active@response
        weights,info=solve_contact_lcp((small+small.T)*.5,free_gap[ids],tolerance=tol,
                                      rank_aware=True,initial=weights,prefer_cholesky=True)
        sim.statistics["contact_restricted_solves"]=sim.statistics.get("contact_restricted_solves",0)+1
        direction=free_direction+response@weights
        test=gaps+G@direction
        worst_active=test[ids].min()
        test[ids]=np.inf
        j=int(np.argmin(test))
        if test[j]>=-10*tol:
            if worst_active < -100*tol:raise RuntimeError("Active contact failed direct feasibility")
            multipliers[ids]=weights
            info.update(outer_working_set_iterations=outer+1,stored_contact_columns=len(ids),
                        all_candidate_minimum_gap=float(min(worst_active,test[j])))
            return direction,multipliers,info
        ids.append(j);weights=np.r_[weights,0.]
    raise RuntimeError("Unresolved contact working set")

