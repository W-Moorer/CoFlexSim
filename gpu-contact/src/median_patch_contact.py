"""Fixed median-dual nodal pressure patches on the original FE surface.

Each triangle is partitioned by edge midpoints and its centroid. The two
subtriangles at each vertex carry one constant-pressure control area. This
changes the pressure space, not the FE displacement space or volume mesh.
"""
import time
import numpy as np
import scipy.sparse as sp
from consistent_mortar_contact import ContactGeometryError, coflex_mortar_native
from area_average_contact import AreaAverageContact, AreaAverageRows


def median_embedding():
    eye=np.eye(3);center=np.ones(3)/3;maps=[];owners=[]
    for i in range(3):
        j=(i+1)%3;k=(i+2)%3
        maps += [np.array([eye[i],(eye[i]+eye[j])/2,center]),
                 np.array([eye[i],center,(eye[i]+eye[k])/2])]
        owners += [i,i]
    return np.asarray(maps),np.asarray(owners)


def voronoi_embedding(tri):
    """Clip a reference triangle into nearest-vertex planar control areas."""
    maps=[];owners=[]
    for i in range(3):
        poly=list(np.eye(3))
        for j in range(3):
            if j==i:continue
            normal=tri[j]-tri[i];mid=(tri[i]+tri[j])/2;out=[]
            for a,b in zip(poly,poly[1:]+poly[:1]):
                da=(a@tri-mid)@normal;db=(b@tri-mid)@normal
                if da<=0:out.append(a)
                if (da<0<db) or (db<0<da):out.append(a+(b-a)*(da/(da-db)))
            poly=out
        for j in range(1,len(poly)-1):
            B=np.array([poly[0],poly[j],poly[j+1]])
            if abs(np.linalg.det(B))>1e-14:maps.append(B);owners.append(i)
    return np.asarray(maps),np.asarray(owners)


class MedianFace:
    def __init__(self,geom,slave,master,fs,fm,owners,mids,s,m,derivatives,gaps_only):
        B,owner=median_embedding();n=len(s)
        p1=getattr(geom,'nodal_P1_projection',False)
        if p1:B=np.repeat(np.eye(3)[None,:,:],3,axis=0);owner=np.arange(3)
        count=len(B)
        parent_all=np.repeat(np.arange(n),count);piece_all=np.tile(np.arange(count),n)
        allB=np.tile(B,(n,1,1));owner_all=owner[piece_all]
        if getattr(geom,'voronoi_support',False):
            maps=[geom.control_maps[slave][j] for j in owners]
            count=9
            parent_all=np.concatenate([np.full(len(a),j,int) for j,(a,b) in enumerate(maps)]) if maps else np.empty(0,int)
            piece_all=np.concatenate([np.arange(len(a)) for a,b in maps]) if maps else np.empty(0,int)
            allB=np.concatenate([a for a,b in maps]) if maps else np.empty((0,3,3))
            owner_all=np.concatenate([b for a,b in maps]) if maps else np.empty(0,int)
        real=np.einsum('nij,njk->nik',allB,s[parent_all])
        embedded=real.copy();height=None;scale=None
        if getattr(geom,'smoothed_projection',False):
            labels=fs[owners[parent_all],owner_all]
            outward=(geom.corner_normals[slave][owners[parent_all],owner_all]
                     if hasattr(geom,'corner_normals') else geom.vertex_normals[slave][labels])
            anchor=s[parent_all,owner_all]
            distance=np.einsum('nki,ni->nk',real-anchor[:,None,:],outward)
            embedded=real-distance[:,:,None]*outward[:,None,:]
            height=-distance
            cross=np.cross(real[:,1]-real[:,0],real[:,2]-real[:,0])
            projected=np.cross(embedded[:,1]-embedded[:,0],embedded[:,2]-embedded[:,0])
            if np.any(np.einsum('ni,ni->n',cross,outward)<=1e-20):
                raise ContactGeometryError('Median patch average normal crosses a sharp feature')
            scale=np.linalg.norm(cross,axis=1)/np.linalg.norm(projected,axis=1)
        repeated=m[parent_all].reshape(-1,9)
        self.native=coflex_mortar_native.SharedFaceBatch(embedded.reshape(-1,9),repeated,geom.radius,
            np.nextafter(float(geom.radius),0.),derivatives,True)
        self.stats=geom.statistics;self.stats['common_overlap_tests']+=self.native.common_prepares
        self.stats['reused_patch_overlaps']+=self.native.near_tests
        self.slave,self.master=slave,master
        self.guard=self.native.values(False);self.weak=self.native.values(True)
        chosen=self.guard['input'];parent=parent_all[chosen];piece=piece_all[chosen]
        self.parent_input=parent
        if height is not None:
            h=height[chosen];factor=scale[chosen]
            # The projected source centroid of each P1 integral gives its
            # exact affine original-height integral. No source geometry is lost.
            tri=embedded[chosen];A=self.weak['area'];I=self.weak['weighted_gap']
            center=(self.weak['points'].reshape(-1,3,3)+I[:,:,None]*self.weak['info'][:,None,:3])/np.maximum(A[:,:,None],1e-300)
            u=tri[:,1]-tri[:,0];v=tri[:,2]-tri[:,0];d=center-tri[:,None,0,:]
            dot=lambda a,b:np.einsum('ni,ni->n',a,b)
            uu,uv,vv=dot(u,u),dot(u,v),dot(v,v);den=uu*vv-uv*uv
            du=np.einsum('nki,ni->nk',d,u);dv=np.einsum('nki,ni->nk',d,v)
            b=(vv[:,None]*du-uv[:,None]*dv)/den[:,None];c=(uu[:,None]*dv-uv[:,None]*du)/den[:,None]
            self.weak['weighted_gap']+=A*((1-b-c)*h[:,0,None]+b*h[:,1,None]+c*h[:,2,None])
            for data in (self.guard,self.weak):
                data['area']*=factor[:,None];data['weighted_gap']*=factor[:,None]
                data['points']*=factor[:,None];data['info'][:,5]*=factor
            tri=embedded[chosen];pts=self.guard['vertex_points'].reshape(-1,12,3)
            u=tri[:,1]-tri[:,0];v=tri[:,2]-tri[:,0];d=pts-tri[:,None,0,:]
            dot=lambda a,b:np.einsum('ni,ni->n',a,b)
            uu,uv,vv=dot(u,u),dot(u,v),dot(v,v);den=uu*vv-uv*uv
            du=np.einsum('nki,ni->nk',d,u);dv=np.einsum('nki,ni->nk',d,v)
            b=(vv[:,None]*du-uv[:,None]*dv)/den[:,None];c=(uu[:,None]*dv-uv[:,None]*du)/den[:,None]
            self.guard['vertex_gaps']+=(1-b-c)*h[:,0,None]+b*h[:,1,None]+c*h[:,2,None]
            mask=np.arange(12)<self.guard['vertex_counts'][:,None]
            low=np.where(mask,self.guard['vertex_gaps'],np.inf).min(1)
            high=np.where(mask,self.guard['vertex_gaps'],-np.inf).max(1)
            for data in (self.guard,self.weak):data['info'][:,3]=low;data['info'][:,4]=high
        if p1:
            mask=np.arange(3)[None,:]==owner_all[chosen,None]
            self.weak['area']*=mask;self.weak['weighted_gap']*=mask
            self.weak['points']*=np.repeat(mask,3,axis=1)
        self.B=allB[chosen];self.owners=owners[parent];self.mids=mids[parent]
        self.nodes=np.c_[fs[self.owners]+geom.offsets[slave],fm[self.mids]+geom.offsets[master]]
        self.cols=(3*self.nodes[:,:,None]+np.arange(3)).reshape(-1,18)
        virtual_face=count*self.owners+piece
        for data in (self.guard,self.weak):
            used=data['samples']>0
            covered=np.bincount(virtual_face[used],data['area'][used].sum(1),minlength=count*len(fs))
            if np.any(covered[virtual_face[used]]>data['info'][used,5]*(1+1e-7)):
                raise ContactGeometryError('Multiple primary sheets in a median pressure patch')
        pair,slot=np.nonzero(np.arange(12)<self.guard['vertex_counts'][:,None])
        self.pair,self.slot=pair,slot
        self.vertex_gaps=self.guard['vertex_gaps'][pair,slot]
        if self.vertex_gaps.min(initial=0.) < -geom.radius:
            raise ContactGeometryError('Median patch overlap exceeds search validity')
        self.wi=np.flatnonzero(self.weak['samples'])
        node=fs[self.owners,owner_all[chosen]]
        self.raw_nodes,self.inv=np.unique(node[self.wi],return_inverse=True)
        aa=self.weak['area'][self.wi].sum(1)
        self.area=np.bincount(self.inv,aa,minlength=len(self.raw_nodes))
        self.keep=np.flatnonzero(self.area>1e-24);self.partials={}
        self.weak_gaps=np.bincount(self.inv,self.weak['weighted_gap'][self.wi].sum(1),minlength=len(self.area))/np.maximum(self.area,1e-300)
        self.offset=geom.offsets[slave]
        if gaps_only:return
        pts=self.weak['points'][self.wi].reshape(-1,3,3).sum(1)
        self.points=np.array([np.bincount(self.inv,pts[:,k],minlength=len(self.area)) for k in range(3)]).T/np.maximum(self.area[:,None],1e-300)
        normals=self.weak['info'][self.wi,:3]
        self.normals=np.array([np.bincount(self.inv,aa*normals[:,k],minlength=len(self.area)) for k in range(3)]).T/np.maximum(self.area[:,None],1e-300)
        self.min_gap=np.full(len(self.area),np.inf);self.max_gap=np.full(len(self.area),-np.inf)
        np.minimum.at(self.min_gap,self.inv,self.weak['info'][self.wi,3]);np.maximum.at(self.max_gap,self.inv,self.weak['info'][self.wi,4])
        self.faces=sp.csr_matrix((aa,(self.inv,self.mids[self.wi])),shape=(len(self.area),len(fm)))

    def record(self,kind,i):
        assert kind=='weak'
        k=self.keep[i];pick=self.wi[self.inv==k];j=pick[np.argmax(self.weak['area'][pick].sum(1))]
        return dict(key=(self.slave,self.master,int(self.raw_nodes[k])),
            ids=np.unique(self.nodes[pick]),normal=self.normals[k],master_face=int(self.mids[j]),
            point=self.points[k],support_area=self.area[k],min_sample_gap=self.min_gap[k],max_sample_gap=self.max_gap[k],
            derivative='fixed_median_dual_total_gap_derivative')

    def triplets(self,kind,ids):
        assert kind=='weak'
        raw=self.keep[ids];mask=np.isin(self.inv,raw);pick=self.wi[mask];nodes=self.inv[mask]
        missing=np.array(sorted(set(pick.tolist())-self.partials.keys()),int)
        if len(missing):self.partials.update(zip(missing,self.native.pressure_partials(missing).reshape(-1,6,18)))
        partial=np.array([self.partials[j] for j in pick])
        d=(partial[:,3:].sum(1)-self.weak_gaps[nodes,None]*partial[:,:3].sum(1))/self.area[nodes,None]
        d[:,:9]=np.einsum('nij,nik->njk',self.B[pick],d[:,:9].reshape(-1,3,3)).reshape(-1,9)
        lut=np.empty(len(self.area),int);lut[raw]=np.arange(len(raw))
        return np.repeat(lut[nodes],18),self.cols[pick].ravel(),d.ravel()

    def directional(self,dx):
        d=np.asarray(dx)[self.cols]
        ds=np.einsum('nij,njk->nik',self.B,d[:,:9].reshape(-1,3,3)).reshape(-1,9)
        _,v=self.native.directional(ds,d[:,9:],self.wi)
        delta=v[:,3:].sum(1)-self.weak_gaps[self.inv]*v[:,:3].sum(1)
        weak=np.bincount(self.inv,delta,minlength=len(self.area))[self.keep]/self.area[self.keep]
        return weak,np.empty(0)


class MedianPatchContact(AreaAverageContact):
    def build(self,x,J,derivatives=True,gaps_only=False):
        tick=time.perf_counter();self.statistics['calls']+=1
        self.statistics['pressure_support_mode']='fixed_median_dual_control_area'
        try:
            faces=[MedianFace(self,*args,derivatives,gaps_only) for args in self.pair_inputs(x)]
            gaps=np.concatenate([f.weak_gaps[f.keep] for f in faces]) if faces else np.empty(0)
            if gaps_only:return gaps
            world=AreaAverageRows(faces,x.size,self.statistics)
            self.statistics['rows_considered']+=len(gaps)
            mapped=world.mapped(J) if derivatives and J.shape[1] else sp.csr_matrix((len(gaps),J.shape[1]))
            return gaps,world,mapped,world
        finally:self.statistics['total_build_seconds']+=time.perf_counter()-tick
