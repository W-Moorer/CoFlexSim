"""Normal-traction mortar work with geometry refreshed in outer iterations.

The contact load is the integral of shape functions times normal pressure.
It is not the total derivative of the moving normalized integration domain.
The inner solve freezes its quadrature/normal coefficients (a quasi-Newton
contact step); final acceptance still checks refreshed gap complementarity
and equilibrium with the same physical normal-traction operator.
"""
import time
import numpy as np
import scipy.sparse as sp
from shared_contact import FacePair
from area_average_contact import AreaAverageContact,AreaAverageRows
from median_patch_contact import MedianFace


def barycentric(points,tri):
    u=tri[:,1]-tri[:,0];v=tri[:,2]-tri[:,0];d=points-tri[:,0]
    dot=lambda a,b:np.einsum('ij,ij->i',a,b)
    uu,uv,vv=dot(u,u),dot(u,v),dot(v,v);den=uu*vv-uv*uv
    b=(vv*dot(d,u)-uv*dot(d,v))/den;c=(uu*dot(d,v)-uv*dot(d,u))/den
    return np.c_[1-b-c,b,c]


class NormalActions:
    def record(self,kind,i):
        result=super().record(kind,i)
        result['derivative']='integrated_normal_virtual_work_frozen_quadrature'
        return result
    def directional(self,dx):
        r,c,v=self.triplets('weak',np.arange(len(self.keep)))
        result=np.bincount(r,v*np.asarray(dx)[c],minlength=len(self.keep))
        return result,np.empty(0)


class NormalMedianFace(NormalActions,MedianFace):
    def __init__(self,*args):
        super().__init__(*args)
        # args: geometry, slave, master, fs, fm, owners, mids, s, m, ...
        selected=self.parent_input
        self.slave_tri=args[7][selected];self.master_tri=args[8][selected]

    def triplets(self,kind,ids):
        assert kind=='weak'
        raw=self.keep[ids];mask=np.isin(self.inv,raw);pick=self.wi[mask];groups=self.inv[mask]
        A=self.weak['area'][pick];area=A.sum(1)
        center=self.weak['points'][pick].reshape(-1,3,3).sum(1)/area[:,None]
        master=barycentric(center,self.master_tri[pick])*area[:,None]
        normal=self.weak['info'][pick,:3]
        gap=self.weak['weighted_gap'][pick].sum(1)/area
        slave=barycentric(center+gap[:,None]*normal,self.slave_tri[pick])*area[:,None]
        coefficients=np.c_[slave,-master]/self.area[groups,None]
        d=(coefficients[:,:,None]*normal[:,None,:]).reshape(-1,18)
        lut=np.empty(len(self.area),int);lut[raw]=np.arange(len(raw))
        return np.repeat(lut[groups],18),self.cols[pick].ravel(),d.ravel()


class NormalP1Face(NormalActions,FacePair):
    def __init__(self,*args):
        super().__init__(*args)
        selected=self.guard['input'];self.slave_tri=args[7][selected];self.master_tri=args[8][selected]

    def triplets(self,kind,ids):
        assert kind=='weak'
        raw=self.keep[ids];mask=np.zeros(len(self.area),bool);mask[raw]=True
        j,k=np.nonzero(mask[self.inv]);pick=self.wi[j];groups=self.inv[j,k]
        a=self.weak['area'][pick,k]
        center=self.weak['points'].reshape(-1,3,3)[pick,k]/a[:,None]
        normal=self.weak['info'][pick,:3]
        mean_gap=self.weak['weighted_gap'][pick,k]/a
        bs=barycentric(center+mean_gap[:,None]*normal,self.slave_tri[pick])
        bm=barycentric(center,self.master_tri[pick])
        coeff=np.c_[bs,-bm]*(a/self.area[groups])[:,None]
        d=(coeff[:,:,None]*normal[:,None,:]).reshape(-1,18)
        lut=np.empty(len(self.area),int);lut[raw]=np.arange(len(raw))
        return np.repeat(lut[groups],18),self.cols[pick].ravel(),d.ravel()


class NormalMortarContact(AreaAverageContact):
    def __init__(self,*args,pressure_basis='P1',**kwargs):
        super().__init__(*args,**kwargs)
        if pressure_basis not in ('P1','median'):raise ValueError(pressure_basis)
        self.pressure_basis=pressure_basis

    def build(self,x,J,derivatives=True,gaps_only=False):
        tick=time.perf_counter();self.statistics['calls']+=1
        self.statistics['force_kind']='integrated_normal_traction'
        self.statistics['coefficient_policy']='our_normal_mortar_not_native_Abaqus'
        self.statistics['contact_linearization']='frozen_quadrature_quasi_Newton'
        self.statistics['pressure_support_mode']=self.pressure_basis
        if getattr(self,'nodal_P1_projection',False):self.statistics['pressure_support_mode']='P1_nodal_normal'
        if getattr(self,'voronoi_support',False):self.statistics['pressure_support_mode']='reference_Voronoi_nodal_control_area'
        cls=NormalMedianFace if self.pressure_basis=='median' else NormalP1Face
        try:
            faces=[cls(self,*args,derivatives,gaps_only) for args in self.pair_inputs(x)]
            gaps=np.concatenate([f.weak_gaps[f.keep] for f in faces]) if faces else np.empty(0)
            if gaps_only:return gaps
            world=AreaAverageRows(faces,x.size,self.statistics)
            self.statistics['rows_considered']+=len(gaps)
            mapped=world.mapped(J) if derivatives and J.shape[1] else sp.csr_matrix((len(gaps),J.shape[1]))
            return gaps,world,mapped,world
        finally:self.statistics['total_build_seconds']+=time.perf_counter()-tick


class SmoothedNormalMortarContact(NormalMortarContact):
    """Median patches project along area-weighted secondary nodal normals.

Flattening is an integration coordinate transform only. Original surface
heights, area measure and displacement interpolation are retained.
"""
    smoothed_projection=True

    def __init__(self,*args,**kwargs):
        kwargs['pressure_basis']='median'
        super().__init__(*args,**kwargs)

    def build(self,x,J,derivatives=True,gaps_only=False):
        self.vertex_normals=[]
        for i,body in enumerate(self.bodies):
            faces=body.spec['contact_faces'];nodes=x[self.offsets[i]:self.offsets[i+1]]
            tri=nodes[faces];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
            normals=np.zeros_like(nodes)
            for j in range(3):
                contribution=cross
                if getattr(self,'angle_weighted_normals',False):
                    u=tri[:,(j+1)%3]-tri[:,j];v=tri[:,(j+2)%3]-tri[:,j]
                    length=np.linalg.norm(cross,axis=1)
                    angle=np.arctan2(length,np.einsum('ij,ij->i',u,v))
                    contribution=cross*(angle/length)[:,None]
                np.add.at(normals,faces[:,j],contribution)
            norm=np.linalg.norm(normals,axis=1);used=np.unique(faces)
            if np.any(norm[used]<1e-20):raise ValueError('Undefined average surface normal')
            normals[used]/=norm[used,None];self.vertex_normals.append(normals)
        self.statistics['projection_normal']=('angle' if getattr(self,'angle_weighted_normals',False) else 'area')+'_weighted_secondary_node'
        return super().build(x,J,derivatives,gaps_only)


class SmoothedP1MortarContact(SmoothedNormalMortarContact):
    """P1 pressure tests, each projected along its nodal averaged normal."""
    nodal_P1_projection=True


class AngleNormalMortarContact(SmoothedNormalMortarContact):
    """Corner-angle weights avoid a planar face's diagonal bias at a vertex."""
    angle_weighted_normals=True


class VoronoiNormalMortarContact(SmoothedNormalMortarContact):
    """Reference nearest-node control regions, with fixed material support."""
    voronoi_support=True

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from median_patch_contact import voronoi_embedding
        self.control_maps=[[voronoi_embedding(t) for t in b.X[b.spec['contact_faces']]] for b in self.bodies]
