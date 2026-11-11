"""Modal dynamics + physical static residual attachments, in a PD local/global loop.

The massless residual is a MODEL reduction, not an exact split of (M/h²+K)^-1.
The nonlinear inertial carrier excludes residual deformation. All scenes use
the same equations, conservative contact cache, and reduced Schur solver.
"""
import sys,time,json,hashlib
from pathlib import Path
from collections import OrderedDict
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from coflex_core import Body,BUILD,DEFAULT_POLICY,rotation
from quasistatic_basis import cached_basis,DEFAULT_BASIS_POLICY
from segment_vertex_contact import SegmentVertexContact
from contact_candidate_cache import ContactCandidateCache
from hard_contact_solver import solve_contact_lcp
from limited_memory_inverse import LimitedMemoryInverse
from consistent_mortar_contact import ContactGeometryError
from pd_time_integration import PDTimeIntegration

sys.path.insert(0,str(BUILD/'quasistatic_native/Release'))
import coflex_pd_native

DEFAULT_PD_POLICY=dict(DEFAULT_BASIS_POLICY,threads=4,absolute_tolerance=1e-8,
    relative_tolerance=1e-5,max_local_iterations=800,max_global_iterations=40,
    gap_tolerance=1e-9,complementarity_tolerance=1e-8,
    candidate_cache_bytes=16*1024**2,contact_workspace_bytes=64*1024**2,contact_broadphase='aabb_bvh',contact_threads=None,exact_projection_cache=True,release_precompute_operators=True,
    contact_skin_fraction=.5,anderson_history=0,reference_tangent_acceleration=True,reference_tangent_steps=1,quasi_newton_history=0,trace_global_iterations=False,contact_rhs_batch=24,
    global_steps_per_projection=1,static_contact_cache_bytes=64*1024**2,regional_compliance_cache_bytes=32*1024**2,open_feature_cosine_floor=None,
    time_integrator='backward_euler',hht_alpha=-.05,reference_tangent_metric='diagonal')


def release_body_precompute(body,native_only=False):
    """Release duplicated setup operators, never the online factors or basis.

    Report logical sparse storage separately from measured process RSS. The
    native release leaves element geometry/material/gradient evaluation intact.
    """
    record=getattr(body,'precompute_release',dict(native_sparse_bytes=0,python_sparse_bytes=0))
    record['native_sparse_bytes']+=int(body.native.release_assembled_operators())
    if not native_only:
        seen=set()
        for name in ('L','K','M','H','metric_stiffness'):
            value=getattr(body,name,None)
            if sp.issparse(value) and id(value) not in seen:
                record['python_sparse_bytes']+=value.data.nbytes+value.indices.nbytes+value.indptr.nbytes;seen.add(id(value))
            setattr(body,name,None)
    body.precompute_release=record


class ReducedBlocks:
    """Body material blocks plus a small dynamic mass update, without zero blocks."""
    def __init__(self,blocks,dynamic=None,mass=None):
        self.blocks=tuple(blocks);self.offsets=np.r_[0,np.cumsum([len(b) for b in blocks])]
        self.shape=(int(self.offsets[-1]),)*2
        self.dynamic=np.asarray([] if dynamic is None else dynamic,int);self.mass=mass

    def extract(self,rows,cols):
        rows=np.asarray(rows);cols=np.asarray(cols);out=np.zeros((len(rows),len(cols)))
        for lo,hi,B in zip(self.offsets[:-1],self.offsets[1:],self.blocks):
            ri=np.flatnonzero((rows>=lo)&(rows<hi));ci=np.flatnonzero((cols>=lo)&(cols<hi))
            out[np.ix_(ri,ci)]=B[np.ix_(rows[ri]-lo,cols[ci]-lo)]
        if self.mass is not None:
            inverse=np.full(self.shape[0],-1,int);inverse[self.dynamic]=np.arange(len(self.dynamic))
            ri=np.flatnonzero(inverse[rows]>=0);ci=np.flatnonzero(inverse[cols]>=0)
            out[np.ix_(ri,ci)]+=self.mass[np.ix_(inverse[rows[ri]],inverse[cols[ci]])]
        return out

    def __matmul__(self,rhs):
        out=np.empty_like(rhs)
        for lo,hi,B in zip(self.offsets[:-1],self.offsets[1:],self.blocks):out[lo:hi]=B@rhs[lo:hi]
        if self.mass is not None:out[self.dynamic]+=self.mass@rhs[self.dynamic]
        return out

    def diagonal(self):
        out=np.concatenate([b.diagonal() for b in self.blocks])
        if self.mass is not None:out[self.dynamic]+=self.mass.diagonal()
        return out

    def with_mass(self,dynamic,mass):return ReducedBlocks(self.blocks,dynamic,mass)

    def with_material(self,material):return ReducedBlocks(material.blocks,self.dynamic,self.mass)


class DiagonalBlock:
    """Energy-scaled reference metric; used only by safeguarded acceleration."""
    def __init__(self,diagonal):self.values=np.asarray(diagonal);self.nbytes=self.values.nbytes
    def __len__(self):return len(self.values)
    def __matmul__(self,rhs):return self.values.reshape((-1,)+(1,)*(rhs.ndim-1))*rhs
    def __getitem__(self,index):
        rows,cols=index;return np.where(rows==cols,self.values[rows],0.)
    def diagonal(self):return self.values.copy()


class ContactMap:
    """Pull sparse contact rows onto traces without a giant volume CSR Jacobian."""
    def __init__(self,blocks,shape):self.blocks=blocks;self.shape=shape
    def restrict_nodes(self,nodes):
        blocks=[]
        for rows,trace,tangent,lo,hi,offset,R in self.blocks:
            keep=np.flatnonzero(np.isin(rows//3,nodes))
            if len(keep):blocks.append((rows[keep],trace[keep],tangent[keep] if offset else None,lo,hi,offset,R))
        return ContactMap(blocks,self.shape)
    def __matmul__(self,direction):
        out=np.zeros((self.shape[0],)+direction.shape[1:])
        for rows,trace,tangent,lo,hi,offset,R in self.blocks:
            value=trace@direction[lo+offset:hi]
            value=self.rotate(value,R)
            if offset:value+=tangent.reshape((-1,)+(1,)*(direction.ndim-1))*direction[lo]
            out[rows]=value
        return out
    def pullback(self,Gx):
        result=np.zeros((Gx.shape[0],self.shape[1]))
        for rows,trace,tangent,lo,hi,offset,R in self.blocks:
            local,material=self.material_weights(Gx,rows,R)
            result[:,lo+offset:hi]=material@trace
            if offset:result[:,lo]=local@tangent
        return result

    @staticmethod
    def material_weights(Gx,rows,R):
        local=Gx[:,rows].tocoo()
        rr=np.repeat(local.row,3);cc=(3*(local.col//3)[:,None]+np.arange(3)).ravel()
        values=(local.data[:,None]*R[local.col%3,:]).ravel()
        material=sp.csr_matrix((values,(rr,cc)),shape=local.shape);material.eliminate_zeros()
        return local,material

    @staticmethod
    def rotate(value,R):
        if value.ndim==1:return (value.reshape(-1,3)@R.T).ravel()
        return np.einsum('ab,nbk->nak',R,value.reshape(-1,3,value.shape[1])).reshape(value.shape)
    def transpose_action(self,world_force):
        out=np.zeros((self.shape[1],)+world_force.shape[1:])
        for rows,trace,tangent,lo,hi,offset,R in self.blocks:
            f=world_force[rows];nonzero=np.flatnonzero(np.any(f!=0.,axis=1) if f.ndim==2 else f!=0.)
            if len(nonzero):
                local=self.rotate(f,R.T)
                used=np.flatnonzero(np.any(local!=0.,axis=1) if local.ndim==2 else local!=0.)
                out[lo+offset:hi]=trace[used].T@local[used]
            if offset:out[lo]=tangent@f
        return out


class ContactRows:
    """All-candidate products are matrix-free; only requested rows materialize."""
    def __init__(self,Gx,mapping,restrict=True):
        self.Gx=Gx
        self.mapping=mapping.restrict_nodes(np.unique(Gx.indices//3)) if restrict else mapping
        self.shape=(Gx.shape[0],mapping.shape[1])
    def __matmul__(self,direction):return self.Gx@(self.mapping@direction)
    def __getitem__(self,rows):return ContactRows(self.Gx[rows],self.mapping,restrict=False)
    def toarray(self):return self.mapping.pullback(self.Gx)

    def small_dense(self):
        """Pull the current rows back once using the already restricted world-space trace."""
        out=np.zeros(self.shape)
        for rows,trace,tangent,lo,hi,offset,R in self.mapping.blocks:
            local=self.Gx[:,rows]
            rotated=self.mapping.rotate(trace,R)
            out[:,lo+offset:hi]=local@rotated
            if offset:out[:,lo]=local@tangent
        return out
    @property
    def T(self):
        owner=self
        class Adjoint:
            def __matmul__(self,weight):return owner.mapping.transpose_action(owner.Gx.T@weight)
        return Adjoint()


def submatrix(A,rows,cols):
    return A.extract(rows,cols) if isinstance(A,ReducedBlocks) else A[np.ix_(rows,cols)]


def project_material_blocks(T,L,K,batch=128):
    """Project unchanged operators without an n-by-s sparse/dense temporary."""
    reduced=np.empty((T.shape[1],T.shape[1]));energy=np.empty(T.shape[1])
    for lo in range(0,T.shape[1],batch):
        hi=min(lo+batch,T.shape[1]);columns=T[:,lo:hi]
        reduced[:,lo:hi]=T.T@(L@columns)
        energy[lo:hi]=np.sum(columns*(K@columns),axis=0)
    return reduced,energy


def cached_material_blocks(T,L,K,folder=None):
    """Cache only reduced offline operators; hash the actual basis and matrices."""
    start=time.perf_counter()
    if folder is None:
        reduced,energy=project_material_blocks(T,L,K)
        return reduced,energy,dict(hit=False,seconds=time.perf_counter()-start)
    digest=hashlib.sha256(b'projected-PD-material-v1')
    for value in (T,L.data,L.indices,L.indptr,K.data,K.indices,K.indptr):
        digest.update(str((value.shape,value.dtype)).encode())
        # Hash rows in bounded chunks, without another copy of the full basis.
        for chunk in np.array_split(value,max(1,(value.nbytes+4194303)//4194304)):
            digest.update(np.ascontiguousarray(chunk).tobytes())
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    key=digest.hexdigest();path=folder/('material_'+key+'.npz');hit=path.exists()
    if hit:
        with np.load(path,allow_pickle=False) as data:reduced=data['L'];energy=data['energy']
        if reduced.shape!=(T.shape[1],)*2 or energy.shape!=(T.shape[1],) or not np.all(np.isfinite(reduced)):
            raise ValueError('Corrupt reduced material cache')
    else:
        reduced,energy=project_material_blocks(T,L,K)
        temporary=path.with_suffix('.pending.npz');np.savez(temporary,L=reduced,energy=energy);temporary.replace(path)
    return reduced,energy,dict(hit=hit,key=key,seconds=time.perf_counter()-start,bytes=path.stat().st_size)


class CondensedSystem:
    """Static block factored once; dynamic Schur updated only with carrier mass."""
    def __init__(self,L,dynamic,static,static_group_sizes=None,contact_cache_bytes=0):
        self.di=np.asarray(dynamic,int);self.zi=np.asarray(static,int)
        self.B=submatrix(L,self.di,self.zi).copy();self.Dfactors=[]
        offsets=np.r_[0,np.cumsum(static_group_sizes if static_group_sizes is not None else [len(self.zi)])]
        if offsets[-1]!=len(self.zi):raise ValueError('Static block partition does not match coordinates')
        for lo,hi in zip(offsets[:-1],offsets[1:]):
            if hi>lo:
                factor=L.diagonal()[self.zi[lo:hi]] if isinstance(L,ReducedBlocks) and all(isinstance(b,DiagonalBlock) for b in L.blocks) else la.cho_factor(submatrix(L,self.zi[lo:hi],self.zi[lo:hi]),check_finite=False)
                if isinstance(factor,np.ndarray) and np.any(factor<=0):raise ValueError('Nonpositive static reference metric')
                self.Dfactors.append((slice(lo,hi),factor))
        self.DB=self.static_solve(self.B.T)
        self.factor=None;self.last_A=None;self.factorizations=0;self.reuses=0
        self.contact_cache=OrderedDict();self.contact_cache_limit=int(contact_cache_bytes)
        self.contact_cache_bytes=0;self.contact_cache_hits=0;self.contact_cache_misses=0
        self.region_cache=OrderedDict();self.region_cache_limit=32*1024**2;self.region_cache_bytes=0
        self.region_hits=0;self.region_misses=0
        self.dynamic_embedding=np.zeros((len(self.di)+len(self.zi),len(self.di)))
        self.dynamic_embedding[self.di]=np.eye(len(self.di));self.dynamic_embedding[self.zi]=-self.DB

    def static_solve(self,rhs):
        result=np.zeros_like(rhs)
        for sl,factor in self.Dfactors:
            result[sl]=rhs[sl]/factor.reshape((-1,)+(1,)*(rhs.ndim-1)) if isinstance(factor,np.ndarray) else la.cho_solve(factor,rhs[sl],check_finite=False)
        return result

    def update(self,H):
        A=submatrix(H,self.di,self.di)-self.B@self.DB
        # Exact value reuse only; do not reuse an approximate inverse after rotation.
        if self.last_A is not None and np.array_equal(A,self.last_A):self.reuses+=1
        else:self.factor=la.cho_factor((A+A.T)*.5,check_finite=False);self.last_A=A.copy();self.factorizations+=1

    def solve(self,rhs):
        vector=rhs.ndim==1;r=np.asarray(rhs).reshape(len(self.di)+len(self.zi),-1)
        z0=self.static_solve(r[self.zi]);a=la.cho_solve(self.factor,r[self.di]-self.B@z0,check_finite=False)
        out=np.empty_like(r);out[self.di]=a;out[self.zi]=z0-self.DB@a
        return out[:,0] if vector else out

    def solve_contact(self,rows):
        """Exact static surface compliance columns, reusable under rigid rotation.

        Cache D^-1 Z_surface^T per material-frame nodal force DOF. Current
        contact weights/normals and the current dynamic Schur remain live.
        """
        rhs=rows.toarray().T
        if not isinstance(rows,ContactRows) or not self.contact_cache_limit:return self.solve(rhs)
        z0=np.zeros((len(self.zi),rhs.shape[1]))
        for nodes,trace,tangent,lo,hi,offset,R in rows.mapping.blocks:
            zi=np.flatnonzero((self.zi>=lo)&(self.zi<hi))
            if not len(zi):continue
            matches=[(sl,factor) for sl,factor in self.Dfactors if sl.start==zi[0] and sl.stop==zi[-1]+1]
            if len(matches)!=1:return self.solve(rhs)
            sl,factor=matches[0]
            if isinstance(factor,np.ndarray):z0[sl]=rhs[self.zi[sl]]/factor[:,None];continue
            _,weights=ContactMap.material_weights(rows.Gx,nodes,R)
            needed=np.unique(weights.indices)
            if not len(needed):continue
            if len(needed)*len(zi)*8>self.contact_cache_limit:return self.solve(rhs)
            keys=[(lo,int(nodes[j])) for j in needed]
            missing=[i for i,key in enumerate(keys) if key not in self.contact_cache]
            self.contact_cache_hits+=len(keys)-len(missing);self.contact_cache_misses+=len(missing)
            # Keep requested entries resident while evicting only old surfaces.
            protected=set(keys);new_bytes=len(missing)*len(zi)*8
            for key in list(self.contact_cache):
                if self.contact_cache_bytes+new_bytes<=self.contact_cache_limit:break
                if key not in protected:self.contact_cache_bytes-=self.contact_cache.pop(key).nbytes
            if missing:
                local_static=self.zi[sl]-lo-offset
                for begin in range(0,len(missing),24):
                    part=missing[begin:begin+24]
                    values=la.cho_solve(factor,trace[np.ix_(needed[part],local_static)].T,check_finite=False)
                    for j,i in enumerate(part):
                        value=values[:,j].copy();self.contact_cache[keys[i]]=value;self.contact_cache_bytes+=value.nbytes
            values=np.column_stack([self.contact_cache[key] for key in keys])
            for key in keys:self.contact_cache.move_to_end(key)
            z0[sl]=(weights[:,needed]@values.T).T
        a=la.cho_solve(self.factor,rhs[self.di]-self.B@z0,check_finite=False)
        out=np.empty_like(rhs);out[self.di]=a;out[self.zi]=z0-self.DB@a
        return out

    def region_get(self,key,compute):
        if key in self.region_cache:
            self.region_hits+=1;self.region_cache.move_to_end(key);return self.region_cache[key]
        self.region_misses+=1;value=compute()
        if value.nbytes<=self.region_cache_limit:
            while self.region_cache and self.region_cache_bytes+value.nbytes>self.region_cache_limit:
                _,old=self.region_cache.popitem(last=False);self.region_cache_bytes-=old.nbytes
            self.region_cache[key]=value;self.region_cache_bytes+=value.nbytes
        return value

    def static_node_responses(self,body_key,nodes,Z,needed,factor):
        rhs=Z[needed].T
        if isinstance(factor,np.ndarray):return rhs/factor[:,None]
        if rhs.nbytes>self.contact_cache_limit:return la.cho_solve(factor,rhs,check_finite=False)
        keys=[(body_key,int(nodes[j])) for j in needed]
        missing=[i for i,key in enumerate(keys) if key not in self.contact_cache]
        self.contact_cache_hits+=len(keys)-len(missing);self.contact_cache_misses+=len(missing)
        protected=set(keys);new_bytes=len(missing)*Z.shape[1]*8
        for key in list(self.contact_cache):
            if self.contact_cache_bytes+new_bytes<=self.contact_cache_limit:break
            if key not in protected:self.contact_cache_bytes-=self.contact_cache.pop(key).nbytes
        if missing:
            values=la.cho_solve(factor,rhs[:,missing],check_finite=False)
            for j,i in enumerate(missing):
                value=values[:,j].copy();self.contact_cache[keys[i]]=value;self.contact_cache_bytes+=value.nbytes
        for key in keys:self.contact_cache.move_to_end(key)
        return np.column_stack([self.contact_cache[key] for key in keys])


class RegionalContactOperator:
    """Contact columns in candidate space; no static-coordinate working set."""
    def __init__(self,system,G):
        self.system=system;self.G=G;self.contexts=[]
        self.E=np.zeros((G.shape[0],len(system.di)))
        for nodes,trace,tangent,lo,hi,offset,R in G.mapping.blocks:
            local,weights=ContactMap.material_weights(G.Gx,nodes,R)
            key=(lo,tuple(nodes.tolist()))
            coupled=system.region_get(('dynamic',key),lambda:trace@system.dynamic_embedding[lo+offset:hi])
            self.E+=weights@coupled
            if offset:self.E[:,np.flatnonzero(system.di==lo)[0]]+=local@tangent
            zi=np.flatnonzero((system.zi>=lo)&(system.zi<hi))
            if len(zi):
                matches=[(sl,factor) for sl,factor in system.Dfactors if sl.start==zi[0] and sl.stop==zi[-1]+1]
                if len(matches)!=1:raise ValueError('Regional static block is not body-local')
                sl,factor=matches[0];Z=trace[:,system.zi[sl]-lo-offset]
                self.contexts.append((key,nodes,Z,weights,factor))

    def columns(self,ids):
        ids=np.asarray(ids,int);s=self.system
        result=self.E@la.cho_solve(s.factor,self.E[ids].T,check_finite=False)
        for key,nodes,Z,weights,factor in self.contexts:
            active=weights[ids];needed=np.unique(active.indices)
            if not len(needed):continue
            # Columns include cross-region response on the same whole body.
            # Their keys refer to material nodes, never old world normals.
            keys=[('static',key,int(nodes[j])) for j in needed]
            missing=[i for i,k in enumerate(keys) if k not in s.region_cache]
            columns=[s.region_get(k,lambda:None) if k in s.region_cache else None for k in keys]
            if missing:
                response=s.static_node_responses(key[0],nodes,Z,needed[missing],factor)
                values=Z@response
                for j,i in enumerate(missing):columns[i]=s.region_get(keys[i],lambda j=j:values[:,j].copy())
            F=np.column_stack(columns)
            result+=(active[:,needed]@(weights@F).T).T
        return result

    def recover(self,multipliers):
        # One coupled recovery after the contact working set has converged.
        return self.system.solve(self.G.T@multipliers)


class QuasiStaticPDSimulator:
    def __init__(self,scene,policy=None,basis_cache=None,contact_discretization='segment_vertex'):
        start=time.perf_counter();self.scene=scene;self.dt=float(scene['dt']);self.policy=DEFAULT_PD_POLICY|dict(policy or {})
        self.time_operator=PDTimeIntegration(self.policy)
        self.volume_mapping_seconds=0.
        if int(self.policy['threads'])<1:raise ValueError('threads must be positive')
        if int(self.policy['reference_tangent_steps'])<1:raise ValueError('reference_tangent_steps must be positive')
        if int(self.policy['quasi_newton_history'])<0:raise ValueError('quasi_newton_history must be nonnegative')
        metric=self.policy['reference_tangent_metric']
        if metric not in ('diagonal','projected_K'):raise ValueError('Unknown reference tangent metric')
        if metric=='projected_K' and self.policy.get('compact_recovery_tolerance'):
            raise ValueError('Projected K metric requires an explicit basis at initialization')
        self.bodies=[]
        self.x_offsets=np.r_[0,np.cumsum([len(spec['vertices']) for spec in scene['bodies']])];self.q_offsets=[0]
        self.surface_envelopes=None
        if self.policy.get('residual_surface_envelopes') is not None:
            from regional_flexibility import surface_envelopes,geometry_bodies
            self.surface_envelopes=surface_envelopes(geometry_bodies(scene['bodies']),scene.get('contact_pairs',[]),
                self.policy['residual_surface_envelopes'],float(scene.get('contact_search_distance',.001)))
        self.material_model=self.policy.get("reduced_material_model","complete_pd")
        if self.material_model not in ("complete_pd","linear","sampled_pd"):raise ValueError("Unknown material model")
        if self.material_model!="complete_pd" and (self.policy.get("compact_recovery_tolerance") or self.policy.get("adaptive_static_enrichment")):
            raise ValueError("Precomputed material operators require a fixed explicit basis")
        if self.material_model=="sampled_pd" and len(scene.get("material_rules",[]))!=len(scene["bodies"]):
            raise ValueError("Each body needs its own material quadrature rule")
        dynamic=[];static=[];blocks=[];reference_blocks=[];q=[]
        for body_index,spec in enumerate(scene['bodies']):
            ready_path=None;ready=None
            if self.policy.get('compact_recovery_tolerance'):
                if basis_cache is None:raise ValueError('Compact recovery needs an offline cache')
                if self.policy.get('static_basis_archive'):raise ValueError('Prepared recovery and external static archive cannot be combined')
                from compact_recovery import runtime_archive,load_runtime,save_runtime
                from regional_flexibility import geometry_bodies,basis_body
                geometry=geometry_bodies([spec])[0];source=geometry
                if self.surface_envelopes is not None:source=basis_body(geometry,self.surface_envelopes[body_index],self.policy.get('principal_frequency_ceiling_hz'))
                ready_path=runtime_archive(geometry,source,self.policy,basis_cache);ready=load_runtime(ready_path)
            # Finish and release each body's setup operators before constructing
            # the next. Do not retain every body's full sparse K/L beside T.
            b=Body(spec,self.dt,DEFAULT_POLICY,False,native_module=coflex_pd_native,assemble_operators=ready is None)
            if self.policy['release_precompute_operators'] and self.material_model=='complete_pd':release_body_precompute(b,native_only=True)
            self.bodies.append(b)
            if b.reference and len(b.fixed):raise ValueError('Use a reference hub or fixed nodes, not both on the same rotating body')
            b.inversion_geometry_cache=[]
            basis_source=b
            if self.surface_envelopes is not None:
                from regional_flexibility import basis_body
                basis_source=basis_body(b,self.surface_envelopes[body_index],self.policy.get('principal_frequency_ceiling_hz'))
            b.contact_nodes=np.unique(b.spec.get('contact_faces',np.empty((0,3),int)))
            b.contact_rows=(3*b.contact_nodes[:,None]+np.arange(3)).ravel()
            recovery=None
            if ready is not None:
                recovery=ready['recovery'];b.Phi=recovery.principal;b.Z=recovery.residual;b.T=recovery
                b.selection=ready['selection'];b.surface_trace=ready['surface_trace'];reduced_L=ready['reduced_L']
                b.reference_column_energy=ready['column_energy'];b.material_cache=ready['material_cache']
            else:
                if self.policy.get('residual_compression_rank'):
                    from residual_compression import cached_compressed_basis
                    b.Phi,b.Z,b.selection=cached_compressed_basis(basis_source,self.policy,basis_cache)
                elif self.policy.get('static_basis_archive'):
                    from residual_compression import load_archive
                    b.Phi,b.Z,b.selection=load_archive(b,self.policy['static_basis_archive'])
                else:b.Phi,b.Z,b.selection=cached_basis(basis_source,self.policy,basis_cache)
                if self.surface_envelopes is not None:b.selection['surface_envelope']=self.surface_envelopes[body_index]
                b.T=np.c_[b.Phi,b.Z];b.Phi=np.ascontiguousarray(b.Phi);b.Z=b.T[:,b.Phi.shape[1]:]
                inverse=np.full(b.X.size,-1,int);inverse[b.free]=np.arange(len(b.free));positions=inverse[b.contact_rows]
                if self.policy.get('compact_recovery_tolerance'):
                    from compact_recovery import cached_recovery,ModalRecovery
                    spatial=b.X.repeat(3,axis=0)[b.free]
                    residual=cached_recovery(b.Z,spatial,positions[positions>=0],basis_cache,
                        float(self.policy['compact_recovery_tolerance']),int(self.policy.get('compact_recovery_block_rows',768)))
                    recovery=ModalRecovery(b.Phi,residual)
                    b.T=recovery.dense();b.Z=residual;b.selection['volume_recovery']=residual.record
                b.surface_trace=np.zeros((len(b.contact_rows),b.T.shape[1]));mask=positions>=0;b.surface_trace[mask]=b.T[positions[mask]]
                reduced_L,b.reference_column_energy,b.material_cache=cached_material_blocks(b.T,b.L,b.K,basis_cache)
                if recovery is not None:save_runtime(ready_path,b,recovery,reduced_L)
            if self.material_model!="complete_pd":
                from reduced_material import ReducedMaterial
                rule=scene["material_rules"][body_index] if self.material_model=="sampled_pd" else None
                b.reduced_material=ReducedMaterial(b,coflex_pd_native,rule)
                if not (self.material_model=="linear" and self.policy.get("linear_pd_metric_diagnostic",False)):
                    reduced_L=b.reduced_material.majorizer
            del basis_source,ready
            b.rd=b.offset+b.Phi.shape[1];b.r=b.offset+b.T.shape[1]
            if metric=='projected_K':
                projected=b.T.T@(b.K@b.T)
                reference_blocks.append(la.block_diag(np.zeros((b.offset,b.offset)),(projected+projected.T)*.5))
            b.carrier_modal_mass=b.Phi.T@(b.mass[b.free,None]*b.Phi)
            b.rotation_cache={}
            b.reduced_L=la.block_diag(np.zeros((b.offset,b.offset)),reduced_L)
            lo=self.q_offsets[-1];self.q_offsets.append(lo+b.r);dynamic.extend(range(lo,lo+b.rd));static.extend(range(lo+b.rd,lo+b.r));blocks.append(b.reduced_L)
            s=la.lstsq(b.T.dense() if hasattr(b.T,'dense') else b.T,b.q[b.offset:],cond=1e-12)[0] if np.any(b.q[b.offset:]) else np.zeros(b.T.shape[1])
            defect=b.T@s-b.q[b.offset:]
            if la.norm(defect)>1e-10+1e-6*la.norm(b.q[b.offset:]):raise ValueError('Initial displacement lies outside the reduced space')
            q.extend(np.r_[b.q[:b.offset],s])
            if recovery is not None:b.T=recovery
            if self.policy['release_precompute_operators']:release_body_precompute(b)
        self.mass=np.concatenate([b.mass for b in self.bodies])
        self.q_offsets=np.asarray(self.q_offsets);self.q=np.asarray(q);self.initial_q=self.q.copy();self.qdot=np.zeros_like(self.q)
        groups=[b.Z.shape[1] for b in self.bodies]
        self.L=ReducedBlocks(blocks);self.condensed=CondensedSystem(self.L,dynamic,static,groups,self.policy['static_contact_cache_bytes'])
        # K-orthogonal construction makes this diagonal analytically. Retain
        # column energies only for the safeguarded reference direction; the
        # true PD material and global cross blocks are unchanged.
        # Arbitrary load-POD rotations need the cross terms for an effective
        # reference direction. This opt-in metric changes only the safeguarded
        # acceleration, not the material energy, mass carrier or contact rule.
        self.K=ReducedBlocks(reference_blocks if metric=='projected_K' else
            [DiagonalBlock(np.r_[np.zeros(b.offset),np.maximum(b.reference_column_energy,0.)]) for b in self.bodies])
        self.tangent_condensed=CondensedSystem(self.K,dynamic,static,groups)
        for system in [self.condensed,self.tangent_condensed]:system.region_cache_limit=int(self.policy['regional_compliance_cache_bytes'])
        self._mapping_cache={};self.mapping_evaluations=0;self.mapping_exact_reuses=0
        self.x,self.J,_=self.mapping(self.q,jacobian='contact');self.carrier,_,_=self.mapping(self.q,carrier=True,jacobian=False);self.Jcarrier=None
        self.carrier_velocity=np.zeros_like(self.x)
        for i,b in enumerate(self.bodies):
            lo,hi=self.q_offsets[i:i+2];sl=slice(self.x_offsets[i],self.x_offsets[i+1]);v=np.asarray(b.spec.get('initial_velocity',np.zeros_like(b.X)),float)
            if b.reference:
                self.qdot[lo]=b.spec.get('initial_angular_velocity' if b.revolute else 'initial_reference_velocity',0.)
                tangent=np.cross(b.axis,self.carrier[sl]-b.center) if b.revolute else np.tile(b.axis,(len(b.X),1))
                self.carrier_velocity[sl]=tangent*self.qdot[lo]
            if la.norm(v)>0:
                _,self.Jcarrier,_=self.mapping(self.q,carrier=True)
                JJ=self.Jcarrier[3*self.x_offsets[i]:3*self.x_offsets[i+1],lo:lo+b.rd].toarray()
                a=la.lstsq(JJ,v.ravel())[0]
                if la.norm(JJ@a-v.ravel())>1e-9+1e-6*la.norm(v):raise ValueError('Initial velocity has unresolved residual inertia')
                self.qdot[lo:lo+b.rd]+=a;self.carrier_velocity[sl]+=(JJ@a).reshape(-1,3)
        self.initial_carrier_velocity=self.carrier_velocity.copy();self.v=self.world_velocity(self.q,self.qdot)
        self.Jcarrier=None;self._mapping_cache[True].pop('carrier',None)
        for b in self.bodies:b.rotation_cache.pop('carrier',None)
        self.t=0.;self.history=[];self.last_contact={};self._geometry_cache=None;self.geometry_exact_reuses=0;self.contact_timings=dict(search=0.,contact_pullback=0.,dual_solve=0.,line_search=0.,local_projection=0.,global_assembly=0.,surface_prediction=0.,material_evaluation=0.,material_gradient_pullback=0.)
        radius=float(scene.get('contact_search_distance',.001));self.candidate_cache=ContactCandidateCache(radius*self.policy['contact_skin_fraction'],self.policy['candidate_cache_bytes'],self.policy['contact_broadphase'])
        if contact_discretization=='segment_vertex':
            self.geometry=SegmentVertexContact(self.bodies,self.x_offsets,scene.get('contact_pairs',[]),radius,geometry_kernel='batched',candidate_cache=self.candidate_cache,open_gap_tolerance=self.policy['gap_tolerance'],open_feature_cosine_floor=self.policy['open_feature_cosine_floor'])
        elif contact_discretization=='mortar_consistent':
            from consistent_mortar_contact import ConsistentMortarContact
            self.geometry=ConsistentMortarContact.for_scene(self.bodies,self.x_offsets,scene,radius,candidate_cache=self.candidate_cache,threads=self.policy['contact_threads'] or self.policy['threads'],gap_rule=self.policy.get('contact_gap_rule','p1_average'))
        else:raise ValueError('Unsupported reduced contact discretization')
        self.contact_discretization=contact_discretization;self.residual_space=None
        self.statistics=dict(online_full_order_factorizations=0,online_full_order_solves=0,local_projection_calls=0,
            reduced_dimension=len(self.q),dynamic_dimension=len(dynamic),algebraic_dimension=len(static),
            full_elastic_dimension=sum(len(b.free) for b in self.bodies),contact_peak_workspace_array_bytes=0,contact_columns_computed=0,contact_column_reuses=0)
        self.statistics.update(inversion_geometry_reuses=0,inversion_geometry_fallbacks=0)
        self.statistics['trial_mapping_reuses']=0
        self.global_modal_selection={'criterion':'physical_K_modes_with_independent_load_response_checks','bodies':[b.selection for b in self.bodies]}
        self.adaptive_correction=None
        if self.policy.get('adaptive_static_enrichment',False):
            from adaptive_correction import AdaptiveCorrection
            self.adaptive_correction=AdaptiveCorrection(self)
        self.preprocess_seconds=time.perf_counter()-start

    def mapping(self,q,carrier=False,jacobian=True):
        cached_mapping=self._mapping_cache.get(carrier)
        reuse=cached_mapping is not None and np.array_equal(q,cached_mapping['q'])
        key=jacobian if jacobian in ('surface','contact') else 'carrier'
        cache_jacobian=bool(jacobian) and (carrier or jacobian in ('surface','contact'))
        if reuse and (not jacobian or (cache_jacobian and key in cached_mapping)):
            self.mapping_exact_reuses+=1
            return cached_mapping['x'],cached_mapping.get(key) if jacobian else None,cached_mapping['local']
        self.mapping_evaluations+=1
        world=[];blocks=[];local=[]
        for i,b in enumerate(self.bodies):
            lo,hi=self.q_offsets[i:i+2];s=q[lo:hi];T=b.Phi if carrier else b.T
            R=rotation(b.axis,s[0]) if b.revolute else np.eye(3)
            if reuse:y=cached_mapping['local'][i];x=cached_mapping['x'][self.x_offsets[i]:self.x_offsets[i+1]]
            else:
                tick=time.perf_counter()
                y=b.X.copy();y.ravel()[b.free]+=T@s[b.offset:b.offset+T.shape[1]]
                x=(y-b.center)@R.T+b.center
                if b.prismatic:x+=s[0]*b.axis
                self.volume_mapping_seconds+=time.perf_counter()-tick
            if jacobian:
                surface=jacobian in ('surface','contact') and not carrier;rows=b.contact_rows if surface else b.free
                data=b.surface_trace if surface else T;theta=float(s[0]) if b.revolute else 0.;kind='surface' if surface else ('carrier' if carrier else 'full')
                if jacobian=='contact' and not carrier:
                    tangent=(np.cross(b.axis,x[b.contact_nodes]-b.center) if b.revolute else np.tile(b.axis,(len(b.contact_nodes),1))).ravel() if b.reference else None
                    blocks.append((rows+3*self.x_offsets[i],data,tangent,lo,hi,b.offset,R))
                    world.append(x);local.append(y);continue
                cached=b.rotation_cache.get(kind)
                if cached is None or cached[0]!=theta:
                    rotated=np.einsum('ab,nbk->nak',R,data.reshape(-1,3,data.shape[1])).reshape(len(rows),data.shape[1])
                    # Store only a surface trace and the small dynamic carrier.
                    if kind!='full':b.rotation_cache[kind]=(theta,rotated)
                else:rotated=cached[1]
                compact=sp.csr_matrix(rotated);indptr=np.zeros(x.size+1,np.int32);indptr[rows+1]=np.diff(compact.indptr);np.cumsum(indptr,out=indptr)
                elastic=sp.csr_matrix((compact.data,compact.indices,indptr),shape=(x.size,b.r-b.offset))
                if b.reference:
                    tangent=(np.cross(b.axis,x-b.center) if b.revolute else np.tile(b.axis,(len(x),1))).ravel()
                    if surface:
                        keep=np.zeros(x.size,bool);keep[b.contact_rows]=True;tangent=np.where(keep,tangent,0.)
                    elastic=sp.hstack((sp.csr_matrix(tangent[:,None]),elastic),format='csr')
                blocks.append(elastic)
            world.append(x);local.append(y)
        x=cached_mapping['x'] if reuse else np.vstack(world)
        if reuse:local=cached_mapping['local']
        J=(ContactMap(blocks,(x.size,len(q))) if jacobian=='contact' and not carrier else sp.block_diag(blocks,format='csr')) if jacobian else None
        if not reuse:cached_mapping={'q':q.copy(),'x':x,'local':local}
        if cache_jacobian:cached_mapping[key]=J
        self._mapping_cache[carrier]=cached_mapping
        return x,J,local

    def world_velocity(self,q,qdot):
        x,_,_=self.mapping(q,jacobian=False);velocity=[]
        for i,b in enumerate(self.bodies):
            lo,hi=self.q_offsets[i:i+2];s=q[lo:hi];rate=qdot[lo:hi];u=np.zeros_like(b.X);u.ravel()[b.free]=b.T@rate[b.offset:]
            R=rotation(b.axis,s[0]) if b.revolute else np.eye(3);v=u@R.T
            if b.reference:
                xx=x[self.x_offsets[i]:self.x_offsets[i+1]];v+=rate[0]*(np.cross(b.axis,xx-b.center) if b.revolute else b.axis)
            velocity.append(v)
        return np.vstack(velocity)

    def carrier_pullback(self,q,force):
        """Analytic inertial adjoint; do not expand the volume modal Jacobian."""
        chi,_,_=self.mapping(q,carrier=True,jacobian=False);result=np.zeros_like(q)
        force=np.asarray(force).reshape(-1,3)
        for i,b in enumerate(self.bodies):
            lo=self.q_offsets[i];sl=slice(self.x_offsets[i],self.x_offsets[i+1])
            R=rotation(b.axis,q[lo]) if b.revolute else np.eye(3)
            result[lo+b.offset:lo+b.rd]=b.Phi.T@(force[sl]@R).ravel()[b.free]
            if b.reference:
                tangent=np.cross(b.axis,chi[sl]-b.center) if b.revolute else np.tile(b.axis,(len(b.X),1))
                result[lo]=np.sum(force[sl]*tangent)
        return result

    def cache_trial_mapping(self,q,local,directions,alpha):
        """Reuse T*d already needed by inversion checks: y_trial=y+alpha*T*d."""
        tick=time.perf_counter();ys=[];xs=[]
        for i,(b,y,dy) in enumerate(zip(self.bodies,local,directions)):
            lo=self.q_offsets[i];trial=y+alpha*dy
            R=rotation(b.axis,q[lo]) if b.revolute else np.eye(3)
            world=(trial-b.center)@R.T+b.center
            if b.prismatic:world+=q[lo]*b.axis
            ys.append(trial);xs.append(world)
        self._mapping_cache[False]={'q':q.copy(),'x':np.vstack(xs),'local':ys}
        self.volume_mapping_seconds+=time.perf_counter()-tick;self.statistics['trial_mapping_reuses']+=1

    def external_gradient(self,q,x,force):
        result=np.zeros_like(q)
        for i,b in enumerate(self.bodies):
            lo,hi=self.q_offsets[i:i+2];sl=slice(self.x_offsets[i],self.x_offsets[i+1]);R=rotation(b.axis,q[lo]) if b.revolute else np.eye(3)
            if not np.any(force[sl]):continue
            result[lo+b.offset:hi]=b.T.T@(force[sl]@R).ravel()[b.free]
            if b.reference:
                tangent=np.cross(b.axis,x[sl]-b.center) if b.revolute else np.tile(b.axis,(len(b.X),1))
                result[lo]=np.sum(force[sl]*tangent)
        return result

    def project(self,q,fields=False,gradient=True):
        if getattr(self,"material_model","complete_pd")!="complete_pd":
            if fields:raise ValueError("Use fields() for postprocessed element observables")
            tick=time.perf_counter();E=0.;g=np.zeros_like(q);minimum=np.inf
            for i,b in enumerate(self.bodies):
                lo,hi=self.q_offsets[i:i+2];s=slice(lo+b.offset,hi)
                out=b.reduced_material.evaluate(q[s],int(self.policy["threads"]))
                E+=out["energy"];g[s]=out["reduced_gradient"];minimum=min(minimum,out["min_det"])
            elapsed=time.perf_counter()-tick
            self.contact_timings["material_evaluation"]+=elapsed;self.contact_timings["local_projection"]+=elapsed
            self.statistics["local_projection_calls"]+=1
            return (q.copy(),E,g,minimum),[]
        tick=time.perf_counter();_,_,local=self.mapping(q,jacobian=False);E=0.;g=np.zeros_like(q);mindet=np.inf;outputs=[]
        for i,b in enumerate(self.bodies):
            native_start=time.perf_counter();cached=getattr(b,'projection_cache',None)
            reusable=not fields and self.policy['exact_projection_cache'] and cached is not None and np.array_equal(local[i],cached['positions'])
            if reusable:
                f=cached['result'];self.statistics['material_projection_exact_reuses']=self.statistics.get('material_projection_exact_reuses',0)+1
            else:
                if self.policy.get('exact_element_projection_cache',False):
                    f=b.native.evaluate_parallel(local[i],fields,int(self.policy['threads']),True)
                else:f=b.native.evaluate_parallel(local[i],fields,int(self.policy['threads']))
                self.statistics['material_projection_evaluations']=self.statistics.get('material_projection_evaluations',0)+1
                if not fields and self.policy['exact_projection_cache']:
                    cached=dict(positions=local[i].copy(),result=f,gradient=None);b.projection_cache=cached
                else:cached=None
            self.contact_timings['material_evaluation']+=time.perf_counter()-native_start
            b.inversion_geometry_cache.append((local[i].copy(),np.asarray(f['minimum_singular_values'])))
            b.inversion_geometry_cache=b.inversion_geometry_cache[-3:]
            outputs.append(f);E+=f['energy'];mindet=min(mindet,f['min_det'])
            if gradient:
                gradient_start=time.perf_counter()
                lo,hi=self.q_offsets[i:i+2]
                value=cached['gradient'] if cached is not None else None
                if value is None:
                    value=b.T.T@np.asarray(f['gradient']).ravel()[b.free]
                    if cached is not None:cached['gradient']=value
                g[lo+b.offset:hi]=value
                self.contact_timings['material_gradient_pullback']+=time.perf_counter()-gradient_start
        self.statistics['local_projection_calls']+=1;self.contact_timings['local_projection']+=time.perf_counter()-tick
        return (q.copy(),E,g if gradient else None,mindet),outputs

    def complete_projection(self,projection,outputs):
        if projection[2] is not None:return projection
        tick=time.perf_counter();g=np.zeros_like(projection[0])
        for i,(b,f) in enumerate(zip(self.bodies,outputs)):
            lo,hi=self.q_offsets[i:i+2];g[lo+b.offset:hi]=b.T.T@np.asarray(f['gradient']).ravel()[b.free]
        elapsed=time.perf_counter()-tick;self.contact_timings['local_projection']+=elapsed;self.contact_timings['material_gradient_pullback']+=elapsed
        return projection[0],projection[1],g,projection[3]

    def material_gradient_world(self,q):
        if getattr(self,"material_model","complete_pd")!="complete_pd":
            rows=[]
            for i,b in enumerate(self.bodies):
                lo,hi=self.q_offsets[i:i+2];R=rotation(b.axis,q[lo]) if b.revolute else np.eye(3)
                rows.append(b.reduced_material.world_gradient(q[lo+b.offset:hi],int(self.policy["threads"]))@R.T)
            return np.vstack(rows)
        _,outputs=self.project(q,gradient=False)
        gradient=[]
        for i,(b,f) in enumerate(zip(self.bodies,outputs)):
            R=rotation(b.axis,q[self.q_offsets[i]]) if b.revolute else np.eye(3)
            gradient.append(np.asarray(f['gradient'])@R.T)
        return np.vstack(gradient)

    def inversion_bound(self,body,positions,direction):
        # Cached singular values are valid only at exactly the projected local
        # positions. Longer fixed-projection inner loops retain the old fallback.
        for cached,sigma in reversed(body.inversion_geometry_cache):
            if np.array_equal(cached,positions):
                self.statistics['inversion_geometry_reuses']+=1
                return body.native.inversion_step_from_singular_values(direction,sigma)
        self.statistics['inversion_geometry_fallbacks']+=1
        return body.native.inversion_step(positions,direction)

    def carrier_terms(self,q,pred):
        """Exact carrier virtual work and small Gauss--Newton mass blocks.

        Rotation commutes with each isotropic lumped nodal mass. Pull residuals
        back to material coordinates; never assemble a mesh-by-state sparse J.
        """
        chi,_,local=self.mapping(q,carrier=True,jacobian=False)
        delta=chi-pred;gradient=np.zeros_like(q);mass=np.zeros((len(self.condensed.di),)*2);cursor=0
        for i,b in enumerate(self.bodies):
            lo=self.q_offsets[i];sl=slice(self.x_offsets[i],self.x_offsets[i+1]);r=b.rd
            R=rotation(b.axis,q[lo]) if b.revolute else np.eye(3)
            material_delta=(delta[sl]@R).ravel();weighted=b.mass*material_delta
            gradient[lo+b.offset:lo+r]=b.Phi.T@weighted[b.free]/self.dt**2
            block=mass[cursor:cursor+r,cursor:cursor+r]
            block[b.offset:,b.offset:]=b.carrier_modal_mass
            if b.reference:
                tangent=(np.cross(b.axis,local[i]-b.center) if b.revolute else np.tile(b.axis,(len(b.X),1))).ravel()
                gradient[lo]=np.dot(tangent,weighted)/self.dt**2
                block[0,0]=np.dot(tangent,b.mass*tangent)
                coupling=b.Phi.T@(b.mass[b.free]*tangent[b.free])
                block[0,1:]=coupling;block[1:,0]=coupling
            cursor+=r
        energy=.5*np.dot(self.mass*delta.ravel(),delta.ravel())/self.dt**2
        factor=self.time_operator.mass_factor
        return energy*factor,gradient*factor,mass/self.dt**2*factor,chi

    def bulk(self,q,pred,force,loads,projection):
        tick=time.perf_counter();x,J,local=self.mapping(q,jacobian='contact');inertia,ig,mass,_=self.carrier_terms(q,pred)
        q0,E0,g0,_=projection;ds=q-q0
        change=self.L@ds if np.any(ds) else np.zeros_like(ds)
        E=E0+g0@ds+.5*ds@change+inertia-float(np.sum(force*x))
        g=g0+change+ig-self.external_gradient(q,x,force)
        for i,b in enumerate(self.bodies):
            if b.reference:E-=loads[i]*q[self.q_offsets[i]];g[self.q_offsets[i]]-=loads[i]
            if b.spec.get('guide_stiffness') or b.spec.get('guide_damping'):
                j=self.q_offsets[i];e,force_guide,k=self.time_operator.guide_terms(b.spec,j,q[j],self.dt)
                idx=np.searchsorted(self.condensed.di,j);mass[idx,idx]+=k;E+=e;g[j]+=force_guide
        H=self.L.with_mass(self.condensed.di,mass)
        scale=np.sqrt(np.maximum(H.diagonal(),1e-20));self.contact_timings['global_assembly']+=time.perf_counter()-tick
        return E,g,H,x,J,local,scale

    def potential(self,q,pred,force,loads,projection):
        x,_,_=self.mapping(q,jacobian=False);chi,_,_=self.mapping(q,carrier=True,jacobian=False);delta=(chi-pred).ravel()
        q0,E0,g0,_=projection;ds=q-q0
        E=E0+(g0@ds+.5*ds@(self.L@ds) if np.any(ds) else 0.)+.5*np.dot(self.mass*delta,delta)/self.dt**2*self.time_operator.mass_factor-float(np.sum(force*x))
        for i,b in enumerate(self.bodies):
            if b.reference:E-=loads[i]*q[self.q_offsets[i]]
            if b.spec.get('guide_stiffness') or b.spec.get('guide_damping'):
                j=self.q_offsets[i];E+=self.time_operator.guide_terms(b.spec,j,q[j],self.dt)[0]
        return E,x

    def constraints(self,x,J,gaps_only=False):
        tick=time.perf_counter()
        try:
            if getattr(self,'surface_envelopes',None) is not None:
                from regional_flexibility import check_envelope
                state=self._mapping_cache[False]
                for i,(b,envelope) in enumerate(zip(self.bodies,self.surface_envelopes)):
                    check_envelope(b,envelope,float(state['q'][self.q_offsets[i]]) if b.reference else 0.,state['local'][i])
            signature=(tuple(hash(b.spec.get('contact_faces',np.empty((0,3),int)).tobytes()) for b in self.bodies),
                tuple(tuple(p) for p in self.geometry.pairs),self.geometry.radius,self.geometry.normal_policy,self.geometry.open_gap_tolerance,self.geometry.open_feature_cosine_floor)
            item=self._geometry_cache
            need_derivatives=J is not None or self.contact_discretization!='mortar_consistent'
            if gaps_only and self.contact_discretization=='mortar_consistent':
                if item is not None and item['signature']==signature and np.array_equal(item['x'],x):
                    self.geometry_exact_reuses+=1
                    return item['result'][0]
                return self.geometry.build_gaps(x)
            if item is not None and item['signature']==signature and np.array_equal(item['x'],x) and (not need_derivatives or item.get('derivatives',True)):
                self.geometry_exact_reuses+=1;result=item['result']
            else:
                kwargs={'derivatives':need_derivatives} if self.contact_discretization=='mortar_consistent' else {}
                result=self.geometry.build(x,sp.csr_matrix((x.size,0)),**kwargs)
                item={'signature':signature,'x':x.copy(),'result':result,'map':None,'derivatives':need_derivatives};self._geometry_cache=item
        except Exception as e:
            state=self._mapping_cache.get(False)
            if state is not None and np.array_equal(state['x'],x):self.failure_q=state['q'].copy()
            else:self.failure_q=self.q.copy()
            self.failure_diagnostics={'geometry_error':str(e),'last_completed_time':self.t};raise
        finally:self.contact_timings['search']+=time.perf_counter()-tick
        tick=time.perf_counter()
        if J is None:G=sp.csr_matrix((len(result[0]),0))
        elif item['map'] is J:G=item['pullback']
        else:
            if hasattr(result[1],'mapped'):
                G=result[1].mapped(J)
            else:
                G=ContactRows(result[1],J) if isinstance(J,ContactMap) else (result[1]@J).tocsr()
            if isinstance(G,ContactRows) and self.policy.get("dense_contact_budget_bytes",0):
                # Account for materialization and one rotated trace. Fall back on large patches.
                needed=8*G.shape[0]*G.shape[1]+sum(trace.nbytes for _,trace,*_ in G.mapping.blocks)
                if needed<=self.policy["dense_contact_budget_bytes"]:
                    G=G.small_dense()
            item['map']=J;item['pullback']=G
        self.contact_timings['contact_pullback']+=time.perf_counter()-tick
        return result[0],result[1],G,result[3]

    def step_qp(self,H,g,G,gaps,records,x,system=None):
        if not self.policy.get("contact_multiplier_warm_start",False):
            return self.qp(H,g,G,gaps,system=system)
        try:
            return self.qp(H,g,G,gaps,system=system,history_context=(records,x))
        except (RuntimeError,la.LinAlgError,MemoryError):
            # A stale or degenerate historical set cannot veto the cold solve.
            self.multiplier_history.stats["cold_fallbacks"]+=1
            return self.qp(H,g,G,gaps,system=system)

    def qp(self,H,g,G,gaps,system=None,history_context=None):
        gap_accuracy=float(self.policy.get('contact_qp_tolerance',1e-12))
        if not np.isfinite(gap_accuracy) or gap_accuracy<=0:
            raise ValueError('Positive finite contact QP gap tolerance required')
        system=self.condensed if system is None else system
        tick=time.perf_counter();system.update(H);d=system.solve(-g);free_gap=gaps+G@d
        if hasattr(G,"world"):
            from shared_contact import solve_working_qp
            out=solve_working_qp(self,system,G,gaps,d,free_gap,gap_accuracy,history_context)
            self.contact_timings['dual_solve']+=time.perf_counter()-tick
            return out
        if not len(gaps) or free_gap.min()>=0.:
            lam=np.zeros(len(gaps));info={'iterations':0,'active_count':0,'interface_compliance_assembly_skipped_for_feasible_free_step':True}
        else:
            # An outer working set bounds dense interface storage. All rows are
            # checked after every restricted solve; no candidate is discarded.
            active=[int(np.argmin(free_gap))];lam=np.zeros(len(gaps));responses=np.empty((len(gaps),0));prefetched={};weights=np.empty(0)
            operator=RegionalContactOperator(system,G) if isinstance(G,ContactRows) and getattr(system,'use_regional_contact',True) else None
            response_columns=operator.columns if operator is not None else lambda ids:G@system.solve_contact(G[ids])
            if isinstance(G,np.ndarray):
                response_columns=lambda ids:G@system.solve(G[ids].T)
            batch=max(1,int(self.policy['contact_rhs_batch']))
            # Batch likely violated rows through the SAME factor. This is only
            # a response cache; active-set membership still follows all gaps.
            candidates=np.flatnonzero(free_gap<0.)
            candidates=candidates[np.argsort(free_gap[candidates])[:batch]]
            def prefetch_bytes(count):return 8*(10*len(system.di)*count+10*count**2+5*len(gaps)+5*len(g)+3*len(gaps)*count)
            initial_weights=None
            if history_context is not None:
                seeds=self.multiplier_history.match(*history_context,free_gap)
                if seeds:
                    worst=active[0]
                    initial_ids=list(seeds)
                    if worst not in seeds:initial_ids.append(worst)
                    count=len(initial_ids)
                    reserve=8*(10*len(system.di)*count+10*count**2+5*len(gaps)+5*len(g)+6*len(gaps)*count)
                    if max(prefetch_bytes(count),reserve)<=self.policy['contact_workspace_bytes']:
                        active=initial_ids;candidates=np.asarray(active,int)
                        initial_weights=np.asarray([seeds.get(i,0.) for i in active])
            while len(candidates)>1 and prefetch_bytes(len(candidates))>self.policy['contact_workspace_bytes']:
                candidates=candidates[:max(1,len(candidates)//2)]
            if prefetch_bytes(len(candidates))>self.policy['contact_workspace_bytes']:raise MemoryError('Active contact workspace exceeds declared memory budget')
            self.statistics['contact_peak_workspace_array_bytes']=max(self.statistics['contact_peak_workspace_array_bytes'],prefetch_bytes(len(candidates)))
            gap_response=response_columns(candidates)
            if initial_weights is not None:
                responses=gap_response[:,:-1]
            for j,index in enumerate(candidates):prefetched[int(index)]=gap_response[:,j]
            self.statistics['contact_columns_computed']+=len(candidates)
            for outer in range(len(gaps)+1):
                ids=np.asarray(active,int);needed=8*(5*len(system.di)*(len(ids)+len(prefetched))+10*len(ids)**2+5*len(gaps)+5*len(g)+3*len(gaps)*(len(ids)+len(prefetched)))
                if needed>self.policy['contact_workspace_bytes']:raise MemoryError('Active contact workspace exceeds declared memory budget')
                self.statistics['contact_peak_workspace_array_bytes']=max(self.statistics['contact_peak_workspace_array_bytes'],needed)
                self.statistics['contact_column_reuses']+=responses.shape[1]
                if active[-1] in prefetched:
                    response=prefetched[active[-1]];self.statistics['contact_column_reuses']+=1
                else:
                    # Refill a SMALL response batch while preserving the exact
                    # one-at-a-time active-set pivot order. Every column is
                    # computed once per QP, and all gaps are checked below.
                    scores=all_gap if outer else free_gap
                    pool=[int(k) for k in np.argsort(scores) if scores[k]<-10*gap_accuracy and int(k) not in prefetched and int(k) not in active]
                    refill=np.asarray([active[-1]]+pool[:batch-1],int)
                    def refill_bytes(count):return 8*(5*len(system.di)*(len(ids)+len(prefetched)+count)+10*len(ids)**2+5*len(gaps)+5*len(g)+3*len(gaps)*(len(ids)+len(prefetched)+count))
                    while len(refill)>1 and refill_bytes(len(refill))>self.policy['contact_workspace_bytes']:refill=refill[:max(1,len(refill)//2)]
                    if refill_bytes(len(refill))>self.policy['contact_workspace_bytes']:
                        response=response_columns(ids[-1:])[:,0];self.statistics['contact_columns_computed']+=1
                    else:
                        values=response_columns(refill)
                        for column,index in enumerate(refill):prefetched[int(index)]=values[:,column]
                        response=prefetched[active[-1]];self.statistics['contact_columns_computed']+=len(refill)
                        self.statistics['contact_response_refill_batches']=self.statistics.get('contact_response_refill_batches',0)+1
                        self.statistics['contact_peak_workspace_array_bytes']=max(self.statistics['contact_peak_workspace_array_bytes'],refill_bytes(0))
                responses=np.c_[responses,response];C=responses[ids]
                # Retain the previous restricted dual solution when adding a
                # contact row. Restarting every nested QP at zero repeats its
                # entire active-set search (81 times on the flat patch).
                weights,info=solve_contact_lcp(C,free_gap[ids],tolerance=gap_accuracy,rank_aware=True,
                    initial=initial_weights if outer==0 and initial_weights is not None else np.r_[weights,0.],prefer_cholesky=True)
                self.statistics['contact_restricted_solves']=self.statistics.get('contact_restricted_solves',0)+1
                all_gap=free_gap+responses@weights
                eligible=all_gap.copy();eligible[ids]=np.inf;j=int(np.argmin(eligible))
                if eligible[j]>=-10*gap_accuracy:
                    lam[ids]=weights
                    corrected=d+(operator.recover(lam) if operator is not None else system.solve(G.T@lam));verified=gaps+G@corrected
                    if np.min(verified,initial=0.) < -100*gap_accuracy:raise RuntimeError('Contact response cache failed direct all-row feasibility check')
                    d=corrected;info.update(outer_working_set_iterations=outer+1,stored_contact_columns=len(ids),stored_static_coordinate_working_set_columns=0,
                        all_candidate_minimum_gap=float(verified.min()),response_cache_direct_gap_difference=float(np.max(abs(verified-all_gap),initial=0.)));break
                active.append(j)
            else:raise RuntimeError('Reduced contact working set failed')
        self.contact_timings['dual_solve']+=time.perf_counter()-tick;return d,lam,info

    def trial_constraints(self,x):
        try:
            if getattr(self,'policy',{}).get('contact_trial_gaps_only',False) and self.contact_discretization=='mortar_consistent':
                return self.constraints(x,None,gaps_only=True)
            return self.constraints(x,None)[0]
        except ContactGeometryError:
            self.statistics['invalid_contact_trials_rejected']=self.statistics.get('invalid_contact_trials_rejected',0)+1
            return None

    def forces(self,t):
        f=np.zeros_like(self.x);loads=[];ramp=min(1.,t/max(self.dt,self.scene.get('load_ramp',self.dt)))
        for i,b in enumerate(self.bodies):
            sl=slice(self.x_offsets[i],self.x_offsets[i+1]);f[sl]=np.asarray(b.spec.get('body_acceleration',[0.,0.,0.]))*b.native.mass[:,None]
            f[sl]+=np.asarray(b.spec.get('nodal_force',np.zeros_like(b.X)))*ramp
            loads.append(b.spec.get('torque' if b.revolute else 'reference_force',0.)*ramp)
        return f,loads

    def prefetch_candidates(self, predicted):
        geometry=self.geometry
        if hasattr(geometry,'surface_positions'):
            predicted=geometry.surface_positions(predicted)
        count=fallbacks=0
        for slave,master in geometry.pairs:
            fs=geometry.bodies[slave].spec['contact_faces']
            fm=geometry.bodies[master].spec['contact_faces']
            if not len(fs) or not len(fm):continue
            a,b=geometry.offsets[slave:slave+2]
            c,d=geometry.offsets[master:master+2]
            rows=self.candidate_cache.query((slave,master),predicted[a:b],fs,predicted[c:d],fm,geometry.radius)
            if rows is None:fallbacks+=1
            else:count+=rows.pair_count
        return count,fallbacks

    def predict_contact_surface(self):
        """Carry actual residual surface deformation; advance only dynamic motion.

        This primes conservative broadphase data, not frozen contact constraints.
        Every trial subsequently refreshes gaps, weights, normals and cache bounds.
        """
        tick=time.perf_counter();predicted=self.x.copy();maximum=0.
        for i,b in enumerate(self.bodies):
            if not len(b.contact_nodes):continue
            lo,hi=self.q_offsets[i:i+2];s=self.q[lo:hi];pred=s.copy()
            pred[:b.rd]+=self.dt*self.qdot[lo:lo+b.rd]
            nodes=self.x_offsets[i]+b.contact_nodes;actual=self.x[nodes]
            R0=rotation(b.axis,s[0]) if b.revolute else np.eye(3)
            R1=rotation(b.axis,pred[0]) if b.revolute else np.eye(3)
            material=(actual-b.center-(s[0]*b.axis if b.prismatic else 0.))@R0+b.center
            increment=b.surface_trace[:,:b.Phi.shape[1]]@(pred[b.offset:b.rd]-s[b.offset:b.rd])
            material+=increment.reshape(-1,3)
            predicted[nodes]=(material-b.center)@R1.T+b.center+(pred[0]*b.axis if b.prismatic else 0.)
            maximum=max(maximum,float(np.max(la.norm(predicted[nodes]-actual,axis=1),initial=0.)))
        count,fallbacks=self.prefetch_candidates(predicted)
        self.contact_timings['surface_prediction']+=time.perf_counter()-tick
        return dict(candidate_triangle_pairs=count,capacity_fallback_pairs=fallbacks,maximum_predicted_surface_motion_m=maximum,
            role='broadphase_prediction_only_geometry_refreshed_at_every_changed_iterate')

    def step(self):
        if self.policy.get("contact_multiplier_warm_start",False) and not hasattr(self,"multiplier_history"):
            if self.scene.get("contact_patch_bound_m") is None:
                raise ValueError("Multiplier history requires bounded local mortar contact")
            from contact_multiplier_history import ContactMultiplierHistory
            self.multiplier_history=ContactMultiplierHistory()
        begin=time.perf_counter();timings_before=self.contact_timings.copy();oldq=self.q.copy();oldx=self.x.copy();oldchi=self.carrier.copy();t=self.t+self.dt
        mapping_before=self.volume_mapping_seconds;prediction=self.predict_contact_surface()
        pred,force,loads=self.time_operator.begin(self,t);q=oldq.copy();reference=None;total_global=0;penalty=float(self.policy.get('merit_penalty_initial',1.));trace=[];aa=[];accelerations=0;tangent_accelerations=0;pending=None;energy_history=[]
        if not np.isfinite(penalty) or penalty<0:raise ValueError('Nonnegative finite initial merit penalty required')
        aa_diagnostics=dict(attempts=0,coefficient_rejections=0,gap_rejections=0,merit_rejections=0)
        secant=LimitedMemoryInverse(self.tangent_condensed,self.policy['quasi_newton_history']) if self.policy['quasi_newton_history'] else None
        for outer in range(int(self.policy['max_local_iterations'])):
            entry=q.copy()
            projection=pending if pending is not None else self.project(q)[0];pending=None
            # A local projection is followed by a global constrained solve of
            # that SAME quadratic material surrogate. Reference inertia is nonlinear.
            moved=False
            for inner in range(int(self.policy['max_global_iterations'])):
                total_global+=1;E,g,H,x,J,local,scale=self.bulk(q,pred,force,loads,projection);gaps,Gx,G,records=self.constraints(x,J)
                if secant is not None and np.array_equal(projection[0],q):
                    secant.update(H.with_material(self.K));secant.observe(q,g)
                d,lam,info=self.step_qp(H,g,G,gaps,records,x)
                accepted=lam.copy();accepted[gaps>self.policy['gap_tolerance']]=0.
                if not getattr(records,'no_open_rows',False):accepted[np.array([r.get('open_direction_only',False) for r in records],bool)]=0.
                residual=float(la.norm((g-G.T@accepted)/scale))
                if reference is None:reference=max(float(la.norm(g/scale)),1e-12)
                threshold=self.policy['absolute_tolerance']+self.policy['relative_tolerance']*reference
                penetration=max(0.,-float(gaps.min())) if len(gaps) else 0.;comp=float(np.max(np.abs(accepted*gaps),initial=0.))
                converged=residual<=threshold and penetration<=self.policy['gap_tolerance'] and comp<=self.policy['complementarity_tolerance']
                if self.policy['trace_global_iterations']:
                    print(json.dumps(dict(kind='global_diagnostic',step=len(self.history)+1,local=outer,global_iteration=inner,residual=residual,threshold=threshold,penetration=penetration,contacts=int(np.count_nonzero(accepted)))),flush=True)
                if converged:lam=accepted;break
                penalty=max(penalty,1.5*float(lam.sum()));violation=np.maximum(-gaps,0.).max(initial=0.);merit=E+penalty*violation;slope=float(g@d-penalty*violation);alpha=1.
                directions=[]
                for i,b in enumerate(self.bodies):
                    lo,hi=self.q_offsets[i:i+2];dx=np.zeros_like(b.X);dx.ravel()[b.free]=b.T@d[lo+b.offset:hi]
                    directions.append(dx)
                    if not hasattr(b,"reduced_material") or not b.reduced_material.safe_step(q[lo+b.offset:hi],d[lo+b.offset:hi]):
                        alpha=min(alpha,self.inversion_bound(b,local[i],dx))
                    if b.revolute:alpha=min(alpha,.025/max(abs(d[lo]),1e-20))
                move=np.max(la.norm((J@d).reshape(-1,3),axis=1));alpha=min(alpha,.4*self.geometry.radius/max(move,1e-20))
                bounded_alpha=alpha
                tick=time.perf_counter();success=False
                for _ in range(30):
                    trial=q+alpha*d;self.cache_trial_mapping(trial,local,directions,alpha)
                    Et,xt=self.potential(trial,pred,force,loads,projection);gt=self.trial_constraints(xt)
                    if gt is None:alpha*=.5;continue
                    if Et+penalty*np.maximum(-gt,0.).max(initial=0.)<=merit+1e-4*alpha*min(slope,0.)+1e-15:
                        q=trial;success=True;trace.append({'kind':'pd','local':outer,'global':inner,'alpha':float(alpha),'bounded_alpha':float(bounded_alpha),'predicted_motion':float(move),'fixed_projection_merit':float(Et+penalty*np.maximum(-gt,0.).max(initial=0.))});break
                    alpha*=.5
                self.contact_timings['line_search']+=time.perf_counter()-tick
                if not success:self.failure_q=q;self.failure_diagnostics={'trace':trace[-20:],'residual':residual};raise RuntimeError('Reduced fixed-projection global line search failed')
                moved=True
                # A successful surrogate-descent step may refresh the local
                # projection immediately. The true projected energy is no
                # greater than its frozen surrogate. Final stationarity and
                # contact acceptance are still checked at a fresh projection.
                limit=self.policy['global_steps_per_projection']
                if limit is not None and inner+1>=int(limit):break
            else:
                self.failure_q=q.copy();self.failure_diagnostics={'trace':trace[-30:],'residual':residual,'threshold':threshold,'outer':outer,'penetration':penetration,'complementarity':comp}
                raise RuntimeError(f'Reduced global solve failed: residual {residual}, threshold {threshold}')
            # At the start of a global solve, the projection gradient is the
            # true PD gradient. If it was already stationary, local/global ends.
            if not moved:
                adaptive=getattr(self,'adaptive_correction',None)
                newq=adaptive.check(q,force,Gx,lam,projection) if adaptive is not None else None
                if newq is None:break
                q=newq;pending=None;reference=None;aa=[]
                secant=LimitedMemoryInverse(self.tangent_condensed,self.policy['quasi_newton_history']) if self.policy['quasi_newton_history'] else None
                continue
            if getattr(self,'single_tangent_sqp',False):
                # The actual-energy SQP branch accepts with the next fresh
                # equilibrium check; it has no PD/Anderson acceleration pass.
                continue
            actual=self.project(q)[0];Eq,gradient,Hq,xq,Jq,ys,_=self.bulk(q,pred,force,loads,actual);gq,_,Gq,rq=self.constraints(xq,Jq)
            true_merit=Eq+penalty*np.maximum(-gq,0.).max(initial=0.);energy_history.append(float(true_merit));pending=actual
            for acceleration_step in range(int(self.policy['reference_tangent_steps']) if self.policy['reference_tangent_acceleration'] else 0):
                if acceleration_step:
                    Eq,gradient,Hq,xq,Jq,ys,_=self.bulk(q,pred,force,loads,pending)
                    gq,_,Gq,rq=self.constraints(xq,Jq)
                metric=Hq.with_material(self.K)
                if secant is not None:secant.update(metric);secant.observe(q,gradient)
                direction,trial_lam,_=self.step_qp(metric,gradient,Gq,gq,rq,xq,system=secant if secant is not None else self.tangent_condensed)
                trial_lam[gq>self.policy['gap_tolerance']]=0.
                if not getattr(rq,'no_open_rows',False):trial_lam[np.array([r.get('open_direction_only',False) for r in rq],bool)]=0.
                # Do not spend trial projections on an already stationary
                # state. The next original PD global check still accepts it.
                if (la.norm((gradient-Gq.T@trial_lam)/np.sqrt(np.maximum(Hq.diagonal(),1e-20)))<=threshold
                    and gq.min(initial=0.)>=-self.policy['gap_tolerance']
                    and np.max(np.abs(trial_lam*gq),initial=0.)<=self.policy['complementarity_tolerance']):break
                move=np.max(la.norm((Jq@direction).reshape(-1,3),axis=1));alpha=min(1.,.4*self.geometry.radius/max(move,1e-20))
                directions=[]
                for i,b in enumerate(self.bodies):
                    lo,hi=self.q_offsets[i:i+2];dx=np.zeros_like(b.X);dx.ravel()[b.free]=b.T@direction[lo+b.offset:hi]
                    directions.append(dx)
                    if not hasattr(b,"reduced_material") or not b.reduced_material.safe_step(q[lo+b.offset:hi],direction[lo+b.offset:hi]):
                        alpha=min(alpha,self.inversion_bound(b,ys[i],dx))
                    if b.revolute:alpha=min(alpha,.025/max(abs(direction[lo]),1e-20))
                tangent_success=False
                bounded_alpha=alpha
                for _ in range(8):
                    trial=q+alpha*direction;self.cache_trial_mapping(trial,ys,directions,alpha)
                    pr,outputs=self.project(trial,gradient=False);Et,xt=self.potential(trial,pred,force,loads,pr);gt=self.trial_constraints(xt)
                    if gt is None:alpha*=.5;continue
                    if pr[3]>0 and Et+penalty*np.maximum(-gt,0.).max(initial=0.)<true_merit:
                        q=trial;pending=self.complete_projection(pr,outputs);true_merit=Et+penalty*np.maximum(-gt,0.).max(initial=0.);tangent_accelerations+=1;tangent_success=True
                        trace.append({'kind':'reference','local':outer,'alpha':float(alpha),'bounded_alpha':float(bounded_alpha),'predicted_motion':float(move),'true_merit':float(true_merit)})
                        break
                    alpha*=.5
                if not tangent_success:break
            # Safeguarded Anderson mixing accelerates the PD fixed point.
            # It does not change the energy, contact model, or stopping test.
            if self.policy['anderson_history']>0:
                aa.append((q.copy(),q-entry));aa=aa[-int(self.policy['anderson_history']):]
                if len(aa)>1:
                    DR=np.column_stack([a[1]-aa[-1][1] for a in aa[:-1]])
                    gamma=la.lstsq(scale[:,None]*DR,-scale*aa[-1][1],cond=1e-10)[0]
                    if la.norm(gamma)<20.:
                        aa_diagnostics['attempts']+=1
                        proposal=q+np.column_stack([a[0]-q for a in aa[:-1]])@gamma;direction=proposal-q
                        move=np.max(la.norm((Jq@direction).reshape(-1,3),axis=1));alpha=min(1.,.4*self.geometry.radius/max(move,1e-20))
                        _,_,ys=self.mapping(q,jacobian=False)
                        directions=[]
                        for i,b in enumerate(self.bodies):
                            lo,hi=self.q_offsets[i:i+2];dx=np.zeros_like(b.X);dx.ravel()[b.free]=b.T@direction[lo+b.offset:hi]
                            directions.append(dx)
                            if not hasattr(b,"reduced_material") or not b.reduced_material.safe_step(q[lo+b.offset:hi],direction[lo+b.offset:hi]):
                                alpha=min(alpha,self.inversion_bound(b,ys[i],dx))
                            if b.revolute:alpha=min(alpha,.025/max(abs(direction[lo]),1e-20))
                        trial=q+alpha*direction;self.cache_trial_mapping(trial,ys,directions,alpha)
                        xt,_,_=self.mapping(trial,jacobian=False);gt=self.trial_constraints(xt)
                        if gt is None:aa_diagnostics['gap_rejections']+=1;continue
                        if gt.min(initial=0.) < -self.policy['gap_tolerance']:
                            aa_diagnostics['gap_rejections']+=1;continue
                        pr,outputs=self.project(trial,gradient=False)
                        Et,xt=self.potential(trial,pred,force,loads,pr);gt=self.trial_constraints(xt)
                        if gt is None:aa_diagnostics['gap_rejections']+=1;continue
                        if pr[3]>0 and gt.min(initial=0.)>=-self.policy['gap_tolerance'] and Et+penalty*np.maximum(-gt,0.).max(initial=0.)<true_merit:
                            q=trial;pending=self.complete_projection(pr,outputs);accelerations+=1
                        else:aa_diagnostics['merit_rejections']+=1
                    else:aa_diagnostics['coefficient_rejections']+=1
        else:
            self.failure_q=q.copy()
            self.failure_diagnostics=dict(residual=residual,threshold=threshold,trace=trace[-30:],
                energy_history=energy_history[-30:],penetration=penetration,complementarity=comp)
            raise RuntimeError(f'PD local/global iteration budget exhausted: {residual}, target {threshold}')
        self.q=q;self.x=x;self.J=J;self.t=t;self.time_operator.finish(self,pred)
        physical_scale=self.time_operator.force_scale
        nodal=np.asarray(Gx.T@(lam*physical_scale)).reshape(-1,3);resultants=[];moments=[];refs=[];angles=[];hub_error=0.;reference_balance=[]
        inertial=self.carrier_pullback(q,self.mass*(self.carrier-pred).ravel()/self.dt**2*self.time_operator.mass_factor)
        force_generalized=self.external_gradient(q,x,force);contact_generalized=np.asarray(G.T@lam).ravel()
        for i,b in enumerate(self.bodies):
            sl=slice(self.x_offsets[i],self.x_offsets[i+1]);resultants.append(nodal[sl].sum(axis=0).tolist());moments.append(np.cross(x[sl]-b.center,nodal[sl]).sum(axis=0).tolist())
            if b.reference:
                theta=float(q[self.q_offsets[i]]);refs.append(theta)
                j=self.q_offsets[i];reference_balance.append(dict(inertial=float(inertial[j]*physical_scale),applied=float((loads[i]+force_generalized[j])*physical_scale),contact=float(contact_generalized[j]*physical_scale),imbalance=float((inertial[j]-loads[i]-force_generalized[j]-contact_generalized[j])*physical_scale),definition='HHT_weighted_nonconstraint_forces' if self.time_operator.scheme=='hht' else 'endpoint_forces'))
                if b.spec.get('guide_stiffness') or b.spec.get('guide_damping'):
                    guide=-physical_scale*self.time_operator.guide_terms(b.spec,j,q[j],self.dt)[1]
                    reference_balance[-1]['guide_force']=guide
                    reference_balance[-1]['imbalance']-=guide
                if b.revolute:angles.append(theta)
                exact=(b.X[b.hub]-b.center)@rotation(b.axis,theta).T+b.center if b.revolute else b.X[b.hub]+theta*b.axis
                hub_error=max(hub_error,float(np.max(np.abs(x[self.x_offsets[i]+b.hub]-exact))))
        impulse,angular_impulse=self.time_operator.contact_impulse(nodal,np.asarray(moments),self.dt)
        endpoint=records.endpoint_columns() if hasattr(records,"endpoint_columns") else None
        if endpoint is None:
            support=np.array([r['support_area'] for r in records])
            normal=lam*physical_scale*np.array([r.get('reaction_scale',1.) for r in records])
            node_ids=[int(r['ids'][0]) for r in records];master_faces=[r['master_face'] for r in records]
            points=[r['point'].tolist() for r in records]
        else:
            support=endpoint["support"];normal=lam*physical_scale
            node_ids=endpoint["node"].tolist();master_faces=endpoint["master"].tolist();points=endpoint["point"].tolist()
        ids=np.flatnonzero(la.norm(nodal,axis=1)>0.)
        self.last_contact=dict(node_ids=node_ids,gap_m=gaps.tolist(),normal_force_N=normal.tolist(),qp_multiplier_N=lam.tolist(),
            lumped_pressure_Pa=(normal/np.maximum(support,1e-30)).tolist(),master_faces=master_faces,support_area_m2=support.tolist(),
            discretization=self.contact_discretization,gap_definition='finite_overlap_vertex_ray_gap',pressure_definition='discrete_force_per_bookkeeping_area_not_point_pressure',
            application_points_m=points,force_node_ids=ids.tolist(),world_nodal_contact_force_N=nodal[ids].tolist(),QP_multiplier_to_physical_force=physical_scale)
        if self.contact_discretization=='mortar_consistent':
            self.last_contact.update(gap_definition='normalized_P1_integral_over_finite_overlap',
                pressure_definition='weak_Lagrange_pressure_coefficient_not_point_CPRESS',
                force_definition='transpose_of_total_normalized_gap_derivative')
            if self.policy.get('contact_gap_rule','p1_average')!='p1_average':
                self.last_contact.update(gap_definition='physical_overlap_vertex_inequalities',
                    pressure_definition='point_constraint_reaction_not_point_CPRESS',force_definition='transpose_of_total_physical_vertex_gap_derivative')
        row=dict(t=t,seconds=time.perf_counter()-begin,iterations=outer+1,global_iterations=total_global,residual=residual,threshold=threshold,
            stationarity_space='retained_modal_and_static_attachment_coordinates',min_det=float(projection[3]),maximum_penetration_m=penetration,complementarity_Nm=comp,
            minimum_candidate_gap_m=float(gaps.min()) if len(gaps) else None,maximum_loaded_gap_m=float(np.max(abs(gaps[lam>1e-8]),initial=0.)),contacts=int(np.count_nonzero(lam>1e-8)),
            contact_resultants=resultants,contact_moments=moments,reference_coordinates=refs,angles=angles,hub_error=hub_error,dual_information=info,basis_dimension=len(q),
            contact_discretization=self.contact_discretization,reported_constraint_gaps_are_weak_averages=self.contact_discretization!='segment_vertex',anderson_accepted=accelerations,anderson_diagnostics=aa_diagnostics,
            quasi_newton=secant.record() if secant is not None else None,
            iteration_diagnostics=trace if self.policy['trace_global_iterations'] else None,
            reference_tangent_accelerations=tangent_accelerations,local_global_merit_last=float(energy_history[-1]) if energy_history else float(E),
            step_timings={key:self.contact_timings[key]-value for key,value in timings_before.items()},
            volume_mapping_seconds=self.volume_mapping_seconds-mapping_before,surface_prediction=prediction,
            reference_generalized_force_balance=reference_balance,time_integration=self.time_operator.record(),
            contact_impulse_resultants_Ns=[impulse[self.x_offsets[i]:self.x_offsets[i+1]].sum(0).tolist() for i in range(len(self.bodies))],
            contact_angular_impulse_Nms=angular_impulse.tolist(),
            reference_velocities=[float(self.qdot[self.q_offsets[i]]) for i,b in enumerate(self.bodies) if b.reference])
        if self.contact_discretization=='mortar_consistent':
            if endpoint is not None:
                row['maximum_loaded_sampled_penetration_m']=float(np.max(-endpoint["minimum"][lam>1e-8],initial=0.))
                row['maximum_loaded_sampled_opening_m']=float(np.max(endpoint["maximum"][lam>1e-8],initial=0.))
            else:
                row['maximum_loaded_sampled_penetration_m']=max([max(0.,-r['min_sample_gap']) for r,l in zip(records,lam) if l>1e-8],default=0.)
                row['maximum_loaded_sampled_opening_m']=max([max(0.,r['max_sample_gap']) for r,l in zip(records,lam) if l>1e-8],default=0.)
            if self.policy.get('contact_gap_rule','p1_average')!='p1_average':
                row['reported_constraint_gaps_are_weak_averages']=False
                row['contact_rule']='physical_overlap_vertex_inequalities'
            if self.scene.get("contact_face_blend_m") is not None:
                row["contact_rule"]="simplex_blended_face_gap"
                row["maximum_physical_facet_penetration_m"]=max(
                    (max(0.,-r["min_sample_gap"]) for r in records),default=0.)
                self.last_contact.update(gap_definition="simplex_blended_face_gap",
                    pressure_definition="face_reaction_divided_by_overlap_area_not_point_pressure",
                    force_definition="transpose_of_blended_gap_derivative")
            if self.scene.get("contact_patch_bound_m") is not None:
                row["contact_rule"]="bounded_local_mortar"
                row["reported_constraint_gaps_are_weak_averages"]=None
                row["contains_weak_average_and_vertex_guard_gaps"]=True
                row["maximum_physical_facet_penetration_m"]=float(np.max(-endpoint["minimum"],initial=0.)) if endpoint is not None else max(
                    (max(0.,-r["min_sample_gap"]) for r in records),default=0.)
                self.last_contact.update(gap_definition="local_P1_means_and_shifted_vertex_guards",
                    constraint_kinds=endpoint["kind"] if endpoint is not None else [r["derivative"] for r in records],
                    pressure_definition="mixed_constraint_reactions_not_point_pressure",
                    force_definition="sum_of_local_mortar_and_vertex_guard_adjoints")
        if getattr(self,'adaptive_correction',None) is not None:
            row['adaptive_correction']=dict(enrichments=self.adaptive_correction.enrichments,
                audit_and_rebuild_seconds=self.adaptive_correction.seconds,
                active_static_dimensions=[len(ids) for ids in self.adaptive_correction.selected],
                representation=self.adaptive_correction.events[-1])
        if self.policy.get("contact_multiplier_warm_start",False):
            self.multiplier_history.commit(records,lam,x)
        if getattr(self,"material_model","complete_pd")!="complete_pd":
            _,_,local=self.mapping(self.q,jacobian=False)
            row["min_det"]=min(b.reduced_material.actual_det(y) for b,y in zip(self.bodies,local))
        self.history.append(row);return row

    def fields(self):
        # Field output does not need a discarded reduced material gradient.
        _,_,local=self.mapping(self.q,jacobian=False)
        if getattr(self,"material_model","complete_pd")=="linear":
            from reduced_material import linear_fields
            return [linear_fields(b,y) for b,y in zip(self.bodies,local)]
        return [b.native.evaluate_parallel(y,True,int(self.policy['threads'])) for b,y in zip(self.bodies,local)]

    def restore_carrier(self,previous_q):
        if self.time_operator.scheme!='backward_euler':raise ValueError('HHT restart requires velocity and acceleration history; legacy position-only restart is unsupported')
        self.carrier,_,_=self.mapping(self.q,carrier=True,jacobian=False)
        previous=self.mapping(previous_q,carrier=True)[0];self.carrier_velocity=(self.carrier-previous)/self.dt

    def backend_record(self):
        return dict(self.statistics,policy=self.policy,time_integration=self.time_operator.record(),material_energy_id=str(coflex_pd_native.energy_id),local_projection_parallelism='OpenMP_elements_deterministic_serial_scatter',openmp_enabled=bool(coflex_pd_native.openmp_enabled),
            released_precompute_logical_storage=[getattr(b,'precompute_release',{}) for b in self.bodies],
            finite_rotation_model='nonlinear_reference_and_principal_inertial_carrier_massless_static_residual',
            reference_tangent_metric=('projected_K_safeguarded_by_actual_PD_merit' if self.policy['reference_tangent_metric']=='projected_K' else 'K_column_energy_diagonal_safeguarded_by_actual_PD_merit'),
            candidate_cache=self.candidate_cache.statistics,dynamic_schur_factorizations=self.condensed.factorizations,dynamic_schur_exact_reuses=self.condensed.reuses,
            stored_basis_bytes=sum(b.T.nbytes for b in self.bodies),material_operator_caches=[b.material_cache for b in self.bodies],
            static_contact_cache=dict(bytes=self.condensed.contact_cache_bytes,limit=self.condensed.contact_cache_limit,
                hits=self.condensed.contact_cache_hits,misses=self.condensed.contact_cache_misses),
            regional_compliance_caches=[dict(bytes=s.region_cache_bytes,limit=s.region_cache_limit,hits=s.region_hits,misses=s.region_misses)
                for s in [self.condensed,self.tangent_condensed]],
            material_block_array_bytes=sum(b.nbytes for b in self.L.blocks)+sum(b.nbytes for b in self.K.blocks),
            material_dense_global_equivalent_bytes=16*len(self.q)**2,
            mapping_evaluations=self.mapping_evaluations,mapping_exact_reuses=self.mapping_exact_reuses,
            geometry_exact_reuses=self.geometry_exact_reuses,
            embedding_and_rotation_cache_bytes=sum(b.surface_trace.nbytes+sum(v[1].nbytes for v in b.rotation_cache.values()) for b in self.bodies),static_factor_dimension=len(self.condensed.zi),
            reference_tangent_factorizations=self.tangent_condensed.factorizations,reference_tangent_exact_reuses=self.tangent_condensed.reuses)
