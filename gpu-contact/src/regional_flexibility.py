"""Conservative offline surface-load envelopes; the online contact mesh is intact.

Declared reference-coordinate intervals and elastic displacement bounds are
inputs, never measured contact trajectories. Triangle bounding balls plus a
Lipschitz bound between reference samples conservatively retain possible pairs.
The physical K/M remain those of the whole constrained body.
"""
import copy
from types import SimpleNamespace
import numpy as np
from scipy.spatial import cKDTree
from coflex_core import rotation


class EnvelopeViolation(RuntimeError):
    pass


def geometry_bodies(specs):
    result=[]
    for spec in specs:
        reference=bool(len(spec.get('hub_nodes',[])))
        kind=spec.get('reference_type','revolute');axis=np.asarray(spec.get('axis',[0.,0.,1.]),float).copy();axis/=np.linalg.norm(axis)
        result.append(SimpleNamespace(X=np.asarray(spec['vertices']),spec=spec,reference=reference,
            revolute=reference and kind=='revolute',prismatic=reference and kind=='prismatic',
            center=np.asarray(spec.get('reference_point',[0.,0.,0.])),axis=axis))
    return result


def surface_envelopes(bodies,pairs,settings,search_radius):
    if len(settings)!=len(bodies):raise ValueError('One declared envelope per body is required')
    data=[];records=[]
    for b,setting in zip(bodies,settings):
        lo,hi=map(float,setting.get('reference_interval',[0.,0.]))
        margin=float(setting['elastic_displacement_bound']);samples=int(setting.get('samples',17))
        if hi<lo or margin<=0 or samples<2:raise ValueError('Invalid motion envelope')
        if not b.reference and (lo!=0 or hi!=0):raise ValueError('Unsupported reference interval')
        faces=b.spec.get('contact_faces',np.empty((0,3),int))
        mid=b.X[faces].mean(axis=1);radii=np.linalg.norm(b.X[faces]-mid[:,None,:],axis=2).max(axis=1,initial=0.)
        poses=[]
        for value in np.linspace(lo,hi,samples):
            R=rotation(b.axis,value) if b.revolute else np.eye(3)
            poses.append((mid-b.center)@R.T+b.center+(value*b.axis if b.prismatic else 0.))
        speed=np.linalg.norm(mid-b.center,axis=1).max(initial=0.) if b.revolute else (1. if b.prismatic else 0.)
        pad=speed*(hi-lo)/(2*(samples-1))
        data.append((np.asarray(poses),radii,pad,margin))
        records.append(dict(reference_interval=[lo,hi],elastic_displacement_bound=margin,samples=samples,
                            unsampled_reference_motion_bound=pad,selected=np.zeros(len(faces),bool)))
    for i,j in pairs:
        for a,b in ((i,j),(j,i)):
            A,ra,pa,ea=data[a];B,rb,pb,eb=data[b]
            if not len(ra) or not len(rb):continue
            distance=cKDTree(B.reshape(-1,3)).query(A.reshape(-1,3))[0].reshape(A.shape[:2])
            records[a]['selected']|=np.any(distance<=ra[None,:]+rb.max()+pa+pb+ea+eb+search_radius,axis=0)
    for b,record in zip(bodies,records):
        faces=b.spec.get('contact_faces',np.empty((0,3),int));selected=record.pop('selected')
        record.update(face_ids=np.flatnonzero(selected).tolist(),nodes=np.unique(faces[selected]).tolist(),
                      total_faces=len(faces),selected_faces=int(selected.sum()),construction='declared_motion_bounding_balls_v1')
    return records


def basis_body(body,envelope,frequency_ceiling=None):
    """Change offline load probes only, not constraints, K/M or online contact."""
    shadow=copy.copy(body);shadow.spec=dict(body.spec)
    shadow.spec['contact_faces']=body.spec['contact_faces'][envelope['face_ids']]
    if not len(shadow.spec['contact_faces']):
        # No reachable contact is a valid load case, but do not silently use
        # the basis builder's all-free-nodes fallback as a claimed reduction.
        raise ValueError('Empty surface envelope: use a declared noncontact basis policy')
    if frequency_ceiling is not None:
        old=np.asarray(body.spec.get('frequency_band_hz',[0.,50.,100.,200.]),float)
        ceiling=float(frequency_ceiling)
        if ceiling<old.max():raise ValueError('Frequency enrichment cannot shrink the declared band')
        shadow.spec['frequency_band_hz']=np.unique(np.r_[old,np.linspace(0.,ceiling,6)]).tolist()
    shadow.spec['residual_surface_envelope']=envelope
    return shadow


def check_envelope(body,record,reference,local):
    lo,hi=record['reference_interval']
    if reference<lo-1e-12 or reference>hi+1e-12:
        raise EnvelopeViolation('Reference motion left the declared offline flexibility envelope')
    displacement=np.linalg.norm(local-body.X,axis=1).max(initial=0.)
    if displacement>record['elastic_displacement_bound']:
        raise EnvelopeViolation('Elastic motion left the declared offline flexibility envelope')
    return float(displacement)
