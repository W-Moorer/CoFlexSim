"""Facet-local median pressure supports, with shared FE displacement nodes.

Split a nodal pressure row by its secondary parent facet. A closed sector can
carry pressure while a separated sector at the same node carries none. This
changes the pressure space; it is not an equivalent acceleration nor a claim
to reproduce proprietary native S2S coefficients. Physical geometry, median
area quadrature, averaged normals and normal virtual work are unchanged.
"""
import time
import numpy as np
import scipy.sparse as sp
from normal_mortar_contact import SmoothedNormalMortarContact,NormalMedianFace
from area_average_contact import AreaAverageRows


class SectorFace(NormalMedianFace):
    def __init__(self,*args):
        super().__init__(*args)
        geom,slave,master,fs,fm=args[:5]
        # Recover each overlap's physical owner before replacing the grouping.
        node=self.raw_nodes[self.inv]
        if getattr(geom,'split_both_features',False):
            secondary=geom.secondary_features[slave];primary=geom.primary_features[master]
            primary_count=int(primary.max()+1);count=int(secondary.max()+1)*primary_count
            key=node*count+secondary[self.owners[self.wi]]*primary_count+primary[self.mids[self.wi]]
        elif hasattr(geom,'secondary_features'):
            labels=geom.secondary_features[slave];count=int(labels.max()+1)
            key=node*count+labels[self.owners[self.wi]]
        elif hasattr(geom,'primary_features'):
            labels=geom.primary_features[master];count=int(labels.max()+1)
            key=node*count+labels[self.mids[self.wi]]
        else:
            count=len(fs);key=node*count+self.owners[self.wi]
        self.raw_nodes,self.inv=np.unique(key,return_inverse=True)
        self.physical_nodes=self.raw_nodes//count;self.facet_ids=self.raw_nodes%count
        self.group_kind=('paired_smooth_components' if getattr(geom,'split_both_features',False) else
                         'secondary_smooth_component' if hasattr(geom,'secondary_features') else
                         'primary_smooth_component' if hasattr(geom,'primary_features') else 'secondary_facet')
        area=self.weak['area'][self.wi].sum(1)
        self.area=np.bincount(self.inv,area,minlength=len(self.raw_nodes))
        self.keep=np.flatnonzero(self.area>1e-24)
        self.weak_gaps=np.bincount(self.inv,self.weak['weighted_gap'][self.wi].sum(1),minlength=len(self.area))/np.maximum(self.area,1e-300)
        points=self.weak['points'][self.wi].reshape(-1,3,3).sum(1)
        self.points=np.array([np.bincount(self.inv,points[:,k],minlength=len(self.area)) for k in range(3)]).T/np.maximum(self.area[:,None],1e-300)
        normals=self.weak['info'][self.wi,:3]
        self.normals=np.array([np.bincount(self.inv,area*normals[:,k],minlength=len(self.area)) for k in range(3)]).T/np.maximum(self.area[:,None],1e-300)
        self.min_gap=np.full(len(self.area),np.inf);self.max_gap=np.full(len(self.area),-np.inf)
        np.minimum.at(self.min_gap,self.inv,self.weak['info'][self.wi,3]);np.maximum.at(self.max_gap,self.inv,self.weak['info'][self.wi,4])
        self.faces=sp.csr_matrix((area,(self.inv,self.mids[self.wi])),shape=(len(self.area),len(fm)))

    def record(self,kind,i):
        r=super().record(kind,i);k=self.keep[i]
        r.update(secondary_physical_node=int(self.physical_nodes[k]),pressure_group=int(self.facet_ids[k]),
                 pressure_space=self.group_kind)
        return r


class SectorRows(AreaAverageRows):
    def endpoint_columns(self):
        r=super().endpoint_columns()
        r['node']=np.concatenate([f.offset+f.physical_nodes[f.keep] for f in self.faces]) if self.faces else np.empty(0,int)
        return r


class SectorPressureContact(SmoothedNormalMortarContact):
    def build(self,x,J,derivatives=True,gaps_only=False):
        tick=time.perf_counter();self.statistics['calls']+=1
        try:
            self.vertex_normals=[]
            for i,body in enumerate(self.bodies):
                faces=body.spec['contact_faces'];nodes=x[self.offsets[i]:self.offsets[i+1]]
                tri=nodes[faces];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normals=np.zeros_like(nodes)
                for j in range(3):np.add.at(normals,faces[:,j],cross)
                length=np.linalg.norm(normals,axis=1);used=np.unique(faces)
                if np.any(length[used]<1e-20):raise ValueError('Undefined average surface normal')
                normals[used]/=length[used,None];self.vertex_normals.append(normals)
            face_type=getattr(self,'pressure_face_type',SectorFace)
            faces=[face_type(self,*args,derivatives,False) for args in self.pair_inputs(x)]
            gaps=np.concatenate([f.weak_gaps[f.keep] for f in faces]) if faces else np.empty(0)
            if gaps_only:return gaps
            world=SectorRows(faces,x.size,self.statistics);self.statistics['rows_considered']+=len(gaps)
            mapped=world.mapped(J) if derivatives and J.shape[1] else sp.csr_matrix((len(gaps),J.shape[1]))
            self.statistics['pressure_support_mode']='primary_feature_median' if hasattr(self,'primary_features') else 'secondary_facet_sector_median'
            return gaps,world,mapped,world
        finally:self.statistics['total_build_seconds']+=time.perf_counter()-tick


def smooth_components(X,faces,angle_degrees=30.):
    """Connected reference facets, split at creases; not a load-dependent fit."""
    tri=X[faces];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(normal,axis=1)
    if np.any(length<1e-20):raise ValueError('Degenerate primary facet')
    normal/=length[:,None];parent=np.arange(len(faces));edges={}
    def root(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    for i,face in enumerate(faces):
        for a,b in zip(face,np.roll(face,-1)):
            key=tuple(sorted((int(a),int(b))))
            if key in edges:
                j=edges[key]
                if normal[i]@normal[j]>np.cos(np.deg2rad(angle_degrees)):parent[root(i)]=root(j)
            else:edges[key]=i
    return np.unique([root(i) for i in range(len(faces))],return_inverse=True)[1]


class FeaturePressureContact(SectorPressureContact):
    """Do not average one pressure constraint across distinct primary features.

All sectors on the same smooth primary component retain their original nodal
average. FE nodes and compliance remain shared; feature splitting only creates
independent nonnegative pressure unknowns, including at a common secondary node.
"""
    def __init__(self,*args,crease_angle_degrees=30.,**kwargs):
        super().__init__(*args,**kwargs)
        self.primary_features=[smooth_components(b.X,b.spec['contact_faces'],crease_angle_degrees) for b in self.bodies]
        self.statistics['reference_crease_angle_degrees']=crease_angle_degrees
        self.statistics['primary_smooth_components']=[int(f.max()+1) for f in self.primary_features]


class SecondaryFeaturePressureContact(SectorPressureContact):
    """Split pressure at secondary creases while sharing elastic displacement.

    Smooth neighboring facets still share one nodal pressure support. A crest
    and flank meeting at a sharp edge may activate independently. This changes
    neither the candidate search nor normal projection; it isolates the effect
    of grouping separated and closed parts into one pressure row.
    """
    def __init__(self,*args,crease_angle_degrees=30.,**kwargs):
        super().__init__(*args,**kwargs)
        self.secondary_features=[smooth_components(b.X,b.spec['contact_faces'],crease_angle_degrees) for b in self.bodies]
        self.statistics['reference_crease_angle_degrees']=crease_angle_degrees
        self.statistics['secondary_smooth_components']=[int(f.max()+1) for f in self.secondary_features]

    def build(self,*args,**kwargs):
        result=super().build(*args,**kwargs)
        self.statistics['pressure_support_mode']='secondary_feature_median'
        return result


class CreaseNormalPressureContact(SecondaryFeaturePressureContact):
    """Feature-local normals accompany feature-local pressure supports.

    Current normals are area averages over the same reference smooth component
    as the pressure row. FE interpolation and the original surface heights
    remain unchanged; no smoothing of the physical corner is performed.
    """
    def build(self,x,J,derivatives=True,gaps_only=False):
        self.corner_normals=[]
        for i,b in enumerate(self.bodies):
            faces=b.spec['contact_faces'];tri=x[self.offsets[i]:self.offsets[i+1]][faces]
            cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
            labels=self.secondary_features[i];count=int(labels.max()+1)
            keys=faces*count+labels[:,None]
            _,inverse=np.unique(keys,return_inverse=True);inverse=inverse.reshape(keys.shape)
            sums=np.zeros((int(inverse.max()+1),3))
            for j in range(3):np.add.at(sums,inverse[:,j],cross)
            norm=np.linalg.norm(sums,axis=1)
            if np.any(norm<1e-20):raise ValueError('Undefined feature-local normal')
            self.corner_normals.append((sums/norm[:,None])[inverse])
        result=super().build(x,J,derivatives,gaps_only)
        self.statistics['projection_normal']='area_weighted_secondary_smooth_component'
        return result


class PairedFeaturePressureContact(CreaseNormalPressureContact):
    """Distinct opposing features have distinct pressure supports.

    One secondary patch can overlap a primary crest and flank during a face
    transition. Sharing their displacement nodes does not require one pressure
    multiplier or one averaged gap. This geometry-based split is independent of
    a reference solution; smooth primary facets still share a pressure row.
    """
    split_both_features=True

    def __init__(self,*args,crease_angle_degrees=30.,**kwargs):
        super().__init__(*args,crease_angle_degrees=crease_angle_degrees,**kwargs)
        self.primary_features=[smooth_components(b.X,b.spec['contact_faces'],crease_angle_degrees) for b in self.bodies]

    def build(self,*args,**kwargs):
        result=super().build(*args,**kwargs)
        self.statistics['pressure_support_mode']='paired_feature_median'
        return result
