"""Geometry-only alternative nodal supports, with crease-consistent normals.

These are opt-in accuracy diagnostics, not claimed Abaqus proprietary formulas.
"""
from sector_pressure_contact import CreaseNormalPressureContact,PairedFeaturePressureContact,SectorRows
from median_patch_contact import voronoi_embedding
import numpy as np
import scipy.sparse as sp


def point_triangle_distance2(p,tri):
    a=tri[:,0];u=tri[:,1]-a;v=tri[:,2]-a;d=p-a
    dot=lambda x,y:np.einsum('ij,ij->i',x,y)
    uu,uv,vv=dot(u,u),dot(u,v),dot(v,v);den=uu*vv-uv*uv
    s=(vv*dot(d,u)-uv*dot(d,v))/den;t=(uu*dot(d,v)-uv*dot(d,u))/den
    distance=np.full(len(tri),np.inf);inside=(s>=0)&(t>=0)&(s+t<=1)
    q=a+s[:,None]*u+t[:,None]*v;distance[inside]=dot(p-q,p-q)[inside]
    for j in range(3):
        start=tri[:,j];edge=tri[:,(j+1)%3]-start
        w=np.clip(dot(p-start,edge)/dot(edge,edge),0.,1.)
        q=start+w[:,None]*edge;distance=np.minimum(distance,dot(p-q,p-q))
    return distance


class AnchoredFeaturePressureContact(PairedFeaturePressureContact):
    """One opposing smooth component per secondary nodal anchor.

Select using closest-point geometry, never the reference force or trajectory.
This isolates whether simultaneous constraints on approaching primary features
overconstrain a nodal patch. It remains an experimental pressure discretization.
"""
    def build(self,x,J,derivatives=True,gaps_only=False):
        _,old,_,_=super().build(x,sp.csr_matrix((x.size,0)),derivatives,False)
        for f in old.faces:
            nprimary=int(self.primary_features[f.master].max()+1);chosen={}
            for k in f.keep:
                node=int(f.physical_nodes[k]);secondary=int(f.facet_ids[k]//nprimary)
                parts=f.wi[f.inv==k];point=x[f.offset+node]
                score=float(point_triangle_distance2(point,f.master_tri[parts]).min())
                key=(node,secondary);candidate=(score,int(k))
                if key not in chosen or candidate<chosen[key]:chosen[key]=candidate
            f.keep=np.array(sorted(v[1] for v in chosen.values()),int)
        gaps=np.concatenate([f.weak_gaps[f.keep] for f in old.faces]) if old.faces else np.empty(0)
        if gaps_only:return gaps
        world=SectorRows(old.faces,x.size,self.statistics)
        mapped=world.mapped(J) if derivatives and J.shape[1] else sp.csr_matrix((len(gaps),J.shape[1]))
        self.statistics['pressure_support_mode']='nearest_primary_feature_per_secondary_anchor'
        return gaps,world,mapped,world


class CreaseVoronoiPressureContact(CreaseNormalPressureContact):
    voronoi_support=True
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.control_maps=[]
        for b in self.bodies:
            maps=[]
            for tri in b.X[b.spec['contact_faces']]:
                B,owner=voronoi_embedding(tri);fractions=np.abs(np.linalg.det(B))
                # Round-off sliver polygons at bisector/vertex coincidences
                # have no resolvable orientation. Bound discarded area.
                keep=fractions>1e-10
                if fractions[~keep].sum()>1e-9:raise ValueError('Unresolved Voronoi area')
                maps.append((B[keep],owner[keep]))
            self.control_maps.append(maps)


class CreaseP1PressureContact(CreaseNormalPressureContact):
    nodal_P1_projection=True


class PairedP1PressureContact(PairedFeaturePressureContact):
    """Continuous nodal P1 weights inside each paired smooth-feature support.

    This changes the pressure test space, not the FE surface geometry. A
    triangle's three weights form a partition of unity, with vanishing weight
    at the opposite edge, instead of the median control patch's sharp cutoff.
    Each feature still activates independently; coefficients are geometry-only.
    """
    nodal_P1_projection=True


class ConnectedPatchPressureContact(CreaseNormalPressureContact):
    """Split disconnected opposing patches, not every primary normal crease.

Connectivity is defined by shared mesh edges among the primary facets present
in a secondary nodal support. No pressure/force threshold or reference motion
is used. Geometry, displacement interpolation and normal work stay unchanged.
"""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.primary_adjacency=[]
        for body in self.bodies:
            faces=body.spec['contact_faces'];adj=[set() for _ in faces];edges={}
            for i,face in enumerate(faces):
                for a,b in zip(face,np.roll(face,-1)):
                    key=tuple(sorted((int(a),int(b))))
                    if key in edges:j=edges[key];adj[i].add(j);adj[j].add(i)
                    else:edges[key]=i
            self.primary_adjacency.append(adj)

    def build(self,x,J,derivatives=True,gaps_only=False):
        _,old,_,_=super().build(x,sp.csr_matrix((x.size,0)),derivatives,False)
        for f in old.faces:
            count=len(self.bodies[f.master].spec['contact_faces']);adj=self.primary_adjacency[f.master]
            old_inv=f.inv.copy();old_nodes=f.physical_nodes.copy();old_features=f.facet_ids.copy()
            nfeature=int(self.secondary_features[f.slave].max()+1)
            keys=np.empty(len(f.wi),np.int64)
            for group in f.keep:
                indices=np.flatnonzero(old_inv==group);mids=f.mids[f.wi[indices]];todo=set(mids.tolist());components={}
                while todo:
                    seed=min(todo);stack=[seed];todo.remove(seed);members=[]
                    while stack:
                        j=stack.pop();members.append(j)
                        for k in sorted(adj[j]&todo):todo.remove(k);stack.append(k)
                    root=min(members)
                    for j in members:components[j]=root
                base=int(old_nodes[group])*nfeature+int(old_features[group])
                keys[indices]=base*count+np.array([components[j] for j in mids])
            f.raw_nodes,f.inv=np.unique(keys,return_inverse=True)
            f.physical_nodes=f.raw_nodes//(nfeature*count);f.facet_ids=f.raw_nodes%(nfeature*count)
            f.group_kind='connected_primary_patch_in_secondary_feature'
            area=f.weak['area'][f.wi].sum(1);f.area=np.bincount(f.inv,area,minlength=len(f.raw_nodes));f.keep=np.flatnonzero(f.area>1e-24)
            f.weak_gaps=np.bincount(f.inv,f.weak['weighted_gap'][f.wi].sum(1),minlength=len(f.area))/np.maximum(f.area,1e-300)
            points=f.weak['points'][f.wi].reshape(-1,3,3).sum(1)
            f.points=np.array([np.bincount(f.inv,points[:,k],minlength=len(f.area)) for k in range(3)]).T/np.maximum(f.area[:,None],1e-300)
            normals=f.weak['info'][f.wi,:3]
            f.normals=np.array([np.bincount(f.inv,area*normals[:,k],minlength=len(f.area)) for k in range(3)]).T/np.maximum(f.area[:,None],1e-300)
            f.min_gap=np.full(len(f.area),np.inf);f.max_gap=np.full(len(f.area),-np.inf)
            np.minimum.at(f.min_gap,f.inv,f.weak['info'][f.wi,3]);np.maximum.at(f.max_gap,f.inv,f.weak['info'][f.wi,4])
            f.faces=sp.csr_matrix((area,(f.inv,f.mids[f.wi])),shape=(len(f.area),count))
        gaps=np.concatenate([f.weak_gaps[f.keep] for f in old.faces]) if old.faces else np.empty(0)
        if gaps_only:return gaps
        world=SectorRows(old.faces,x.size,self.statistics)
        mapped=world.mapped(J) if derivatives and J.shape[1] else sp.csr_matrix((len(gaps),J.shape[1]))
        self.statistics['pressure_support_mode']='connected_primary_patch_in_secondary_feature'
        return gaps,world,mapped,world
