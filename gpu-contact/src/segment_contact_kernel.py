"""Compiled finite-overlap geometry, equivalent to the Python vertex rule.

Scene-independent numeric kernel only. The Python reference remains available
for parity tests; this adds no approximate contact or altered physical tolerance.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def cross(a,b):
    return np.array([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])


@njit(cache=True)
def bary(points,tri):
    a=tri[1]-tri[0];b=tri[2]-tri[0];det=a[0]*b[1]-a[1]*b[0]
    out=np.empty((len(points),3))
    for i in range(len(points)):
        p=points[i]-tri[0];u=(p[0]*b[1]-p[1]*b[0])/det;v=(a[0]*p[1]-a[1]*p[0])/det
        out[i,0]=1-u-v;out[i,1]=u;out[i,2]=v
    return out


@njit(cache=True)
def clip(poly,c):
    if len(poly)<3:return np.empty((0,2))
    result=np.empty((len(poly)+2,2));count=0;previous=poly[-1];fp=previous@c[:2]+c[2]
    for current in poly:
        fc=current@c[:2]+c[2]
        if (fp>=-2e-13)!=(fc>=-2e-13):
            result[count]=previous+(current-previous)*(fp/(fp-fc));count+=1
        if fc>=-2e-13:result[count]=current;count+=1
        previous=current;fp=fc
    if count<3:return np.empty((0,2))
    return result[:count].copy()


@njit(cache=True)
def vertex_rule(secondary,main,radius):
    empty=(np.empty((0,3)),np.empty((0,3)),np.empty(0),np.empty(0),np.zeros(3),0.)
    edge=secondary[1]-secondary[0];length=np.sqrt(edge@edge)
    n=cross(edge,secondary[2]-secondary[0]);area2=np.sqrt(n@n)
    if length<=0 or area2<=1e-24:return empty
    outward=n/area2;normal=-outward
    em=cross(main[1]-main[0],main[2]-main[0]);am=np.sqrt(em@em)
    if am<=1e-24 or (em/am)@outward>-.25:return empty
    e1=edge/length;e2=cross(outward,e1)
    S=np.empty((3,2));M=np.empty((3,2))
    for i in range(3):
        for j in range(2):
            axis=e1 if j==0 else e2
            S[i,j]=(secondary[i]-secondary[0])@axis/length
            M[i,j]=(main[i]-secondary[0])@axis/length
    a=M[1]-M[0];b=M[2]-M[0]
    if abs(a[0]*b[1]-a[1]*b[0])<1e-12:return empty
    probes=np.array([[0.,0.],[1.,0.],[0.,1.]]);bm=bary(probes,M)
    poly=S.copy()
    for j in range(3):
        poly=clip(poly,np.array([bm[1,j]-bm[0,j],bm[2,j]-bm[0,j],bm[0,j]]))
    if len(poly)<3:return empty
    gaps=np.empty(3)
    for i in range(3):
        point=secondary[0]+probes[i,0]*length*e1+probes[i,1]*length*e2
        gaps[i]=(point-bm[i]@main)@normal/length
    gc=np.array([gaps[1]-gaps[0],gaps[2]-gaps[0],gaps[0]])
    for sign in (-1.,1.):
        c=sign*gc;c[2]+=radius/length;poly=clip(poly,c)
    if len(poly)<3:return empty
    unique=np.empty_like(poly);count=0
    for p in poly:
        duplicate=False
        for i in range(count):
            delta=p-unique[i]
            if np.sqrt(delta@delta)<1e-12:duplicate=True;break
        if not duplicate:unique[count]=p;count+=1
    poly=unique[:count].copy()
    if len(poly)<3:return empty
    weights=np.zeros(count)
    for i in range(1,count-1):
        a=poly[i]-poly[0];b=poly[i+1]-poly[0]
        area=.5*abs(a[0]*b[1]-a[1]*b[0])*length**2
        weights[0]+=area/3;weights[i]+=area/3;weights[i+1]+=area/3
    if weights.sum()<area2*1e-14:return empty
    Ns=bary(poly,S);Nm=bary(poly,M);g=(Ns@secondary-Nm@main)@normal
    return Ns,Nm,weights,g,normal,area2/2


def compiled_overlap_vertices(secondary,main,radius):
    result=vertex_rule(secondary,main,radius)
    return result if len(result[2]) else None


@njit(cache=True)
def compiled_overlap_batch(secondary,triangles,candidates,radius):
    """Preserve pair order/rules, crossing the Python boundary once per face."""
    result=[]
    for index in candidates:
        rule=vertex_rule(secondary,triangles[index],radius)
        if len(rule[2]):result.append((index,rule))
    return result


@njit(cache=True)
def packed_overlap_batch(secondary,triangles,candidates,radius,face,master_faces,slave_offset,master_offset,open_tolerance,feature,open_feature_floor=-1.):
    """Same finite-feature rows; move per-vertex numeric allocations out of Python."""
    rows=[];coverage=0.;face_area=0.;segments=0;raw=0;cut=0;opened=0;maximum_scale=1.
    for m in candidates:
        Ns,Nm,w,g,ray,area=vertex_rule(secondary,triangles[m],radius)
        if not len(w):continue
        face_area=area;coverage+=w.sum();segments+=1;raw+=len(w)
        ids0=np.concatenate((face+slave_offset,master_faces[m]+master_offset))
        for k in range(len(w)):
            if abs(g[k]-radius)<=max(1e-13,radius*1e-8):cut+=1;continue
            if abs(g[k]+radius)<=max(1e-13,radius*1e-8):raise RuntimeError('Penetration reached the contact search horizon')
            is_open=g[k]>open_tolerance;can_fallback=is_open and open_feature_floor>=0.;normal=ray.copy();scale=1.
            if feature and (not is_open or open_feature_floor>=0.):
                si=np.flatnonzero(Ns[k]>1e-8);mi=np.flatnonzero(Nm[k]>1e-8)
                if len(si)==1:normal=cross(triangles[m,1]-triangles[m,0],triangles[m,2]-triangles[m,0])
                elif len(mi)==1:normal=ray.copy()
                elif len(si)==2 and len(mi)==2:normal=cross(secondary[si[1]]-secondary[si[0]],triangles[m,mi[1]]-triangles[m,mi[0]])
                else:raise RuntimeError('Overlap vertex has no vertex/face or edge/edge feature')
                norm=np.linalg.norm(normal)
                if norm<1e-24:
                    if not can_fallback:raise RuntimeError('Degenerate contact feature normal')
                    normal=ray.copy();is_open=True
                else:
                    normal=normal/norm
                    if normal@ray<0:normal=-normal
                    cosine=normal@ray
                    if can_fallback and cosine<open_feature_floor:normal=ray.copy();is_open=True
                    else:
                        if cosine<1e-8:raise RuntimeError('Contact feature is singular along the selected ray')
                        normal=normal/cosine;scale=1./cosine;is_open=False
            opened+=int(is_open)
            maximum_scale=max(maximum_scale,scale)
            shape=np.concatenate((Ns[k],-Nm[k]));keep=np.flatnonzero(np.abs(shape)>1e-13)
            ids=ids0[keep];vec=shape[keep,None]*normal;order=np.argsort(ids);ids=ids[order];vec=vec[order]
            rounded=np.round(vec.ravel(),11)
            rows.append((m,ids,vec,rounded,normal,scale,Nm[k]@triangles[m],w[k],g[k],is_open,int(face[np.argmax(Ns[k])]+slave_offset)))
    if face_area and coverage/face_area>1.+1e-7:raise RuntimeError('Multiple main sheets require a visibility partition')
    return rows,segments,raw,cut,opened,maximum_scale,coverage/face_area if face_area else 0.
