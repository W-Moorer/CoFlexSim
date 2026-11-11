"""A normal-ray anchor selects the opposing feature of an original nodal patch.

Geometry-only, opt-in contact model diagnostic. Select the nearest intersection
of the secondary anchor's projection line with a candidate main facet; use the
closest finite-triangle distance only when the line misses every main feature.
The selected region retains median pressure weights, feature-local normals,
and integrated normal virtual work. No imposed ideal gear ratio or fitted load.
"""
import numpy as np
import scipy.sparse as sp
from sector_pressure_contact import PairedFeaturePressureContact,SectorRows
from pressure_support_diagnostics import point_triangle_distance2,AnchoredFeaturePressureContact


class AnchoredP1PressureContact(AnchoredFeaturePressureContact):
    """One opposing feature per anchor with continuous P1 pressure weights.

    Uses the existing lumped enforcement compliance. This is an independent
    pressure-discretization candidate, not a consistent pressure-mass variant.
    """
    nodal_P1_projection=True


def normal_ray_distance(point,normal,tri):
    u=tri[:,1]-tri[:,0];v=tri[:,2]-tri[:,0]
    plane=np.cross(u,v);den=plane@normal
    scale=np.linalg.norm(plane,axis=1)*np.linalg.norm(normal)
    valid=abs(den)>1e-12*scale
    distance=np.divide(np.einsum('ij,ij->i',point-tri[:,0],plane),den,
        out=np.zeros(len(tri)),where=valid)
    projected=point-distance[:,None]*normal
    d=projected-tri[:,0]
    dot=lambda a,b:np.einsum('ij,ij->i',a,b)
    uu,uv,vv=dot(u,u),dot(u,v),dot(v,v);det=uu*vv-uv*uv
    b=(vv*dot(d,u)-uv*dot(d,v))/det;c=(uu*dot(d,v)-uv*dot(d,u))/det
    inside=valid&(b>=-1e-10)&(c>=-1e-10)&(b+c<=1+1e-10)
    return np.where(inside,abs(distance)*np.linalg.norm(normal),np.inf)


class NormalAnchoredFeaturePressureContact(PairedFeaturePressureContact):
    def build(self,x,J,derivatives=True,gaps_only=False):
        _,old,_,_=super().build(x,sp.csr_matrix((x.size,0)),derivatives,False)
        chosen_by_ray=fallbacks=changed=0
        for f in old.faces:
            nprimary=int(self.primary_features[f.master].max()+1);groups={}
            for k in f.keep:
                node=int(f.physical_nodes[k]);secondary=int(f.facet_ids[k]//nprimary)
                parts=f.wi[f.inv==k];point=x[f.offset+node]
                ray=float(normal_ray_distance(point,f.normals[k],f.master_tri[parts]).min())
                euclidean=float(point_triangle_distance2(point,f.master_tri[parts]).min())
                groups.setdefault((node,secondary),[]).append((ray,euclidean,int(k)))
            chosen=[]
            for values in groups.values():
                nearest=min(values,key=lambda a:(a[1],a[2]))
                hits=[a for a in values if np.isfinite(a[0])]
                if hits:
                    pick=min(hits,key=lambda a:(a[0],a[1],a[2]));chosen_by_ray+=1
                else:pick=nearest;fallbacks+=1
                changed+=pick[2]!=nearest[2];chosen.append(pick[2])
            f.keep=np.array(sorted(chosen),int)
        gaps=np.concatenate([f.weak_gaps[f.keep] for f in old.faces]) if old.faces else np.empty(0)
        self.statistics.update(pressure_support_mode='normal_ray_primary_feature_per_secondary_anchor',
            last_anchor_ray_rows=chosen_by_ray,last_anchor_fallback_rows=fallbacks,last_anchor_changed_rows=changed)
        if gaps_only:return gaps
        world=SectorRows(old.faces,x.size,self.statistics)
        mapped=world.mapped(J) if derivatives and J.shape[1] else sp.csr_matrix((len(gaps),J.shape[1]))
        return gaps,world,mapped,world
