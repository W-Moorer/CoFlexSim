"""Warm-start bounded mortar with endpoint QP multipliers from the last accepted step."""
import time
import numpy as np


def feature_mask(triangles, points):
    """Barycentric support of the projected point in each physical triangle."""
    u=triangles[:,1]-triangles[:,0];v=triangles[:,2]-triangles[:,0]
    p=points-triangles[:,0]
    dot=lambda a,b: np.einsum("ij,ij->i",a,b)
    uu,uv,vv=dot(u,u),dot(u,v),dot(v,v)
    det=uu*vv-uv*uv
    valid=det>1e-12*uu*vv
    det=np.where(valid,det,1.)
    b=(vv*dot(p,u)-uv*dot(p,v))/det
    c=(uu*dot(p,v)-uv*dot(p,u))/det
    bary=np.column_stack([1.-b-c,b,c])
    valid &= (bary.min(axis=1)>=-1e-7)
    return np.where(valid,(bary>1e-7)@np.array([1,2,4]),0)


def identities(records,indices,x):
    # The last digit of a guard key is a clipping polygon slot. A slot can
    # change identity as faces slide. Verify both parent features as well.
    result={}
    guards=[]
    for i in indices:
        key=records[i]["key"]
        if len(key)==3:
            result[i]=key
        elif len(key)==5:
            guards.append(i)
    if guards:
        nodes=np.array([records[i]["ids"] for i in guards])
        points=np.array([records[i]["point"] for i in guards])
        slave=feature_mask(x[nodes[:,:3]],points)
        main=feature_mask(x[nodes[:,3:]],points)
        for i,a,b in zip(guards,slave,main):
            if a and b and (a!=7 or b!=7):
                result[i]=records[i]["key"]+(int(a),int(b))
    return result


class ContactMultiplierHistory:
    def __init__(self):
        self.forces={}
        self.row_keys=set()
        self.stats=dict(matching_seconds=0.,commit_seconds=0.,lookup_qps=0,matched_qps=0,
                        matched_rows=0,eligible_history_rows=0,cold_fallbacks=0)

    def commit(self,records,lam,x):
        start=time.perf_counter()
        ids=np.flatnonzero(lam>1e-8)
        keys=identities(records,ids,x)
        self.forces={key:float(lam[i]) for i,key in keys.items()}
        self.row_keys={records[i]["key"] for i in keys}
        self.stats["commit_seconds"]+=time.perf_counter()-start

    def match(self,records,x,free_gap):
        start=time.perf_counter()
        self.stats["lookup_qps"]+=1
        self.stats["eligible_history_rows"]+=len(self.forces)
        matched={}
        if self.row_keys:
            if hasattr(records,"matching_indices"):
                ids=sorted(i for i in records.matching_indices(self.row_keys) if free_gap[i]<0.)
            else:ids=[i for i,r in enumerate(records) if r["key"] in self.row_keys and free_gap[i]<0.]
            for i,key in identities(records,ids,x).items():
                if key in self.forces:
                    matched[i]=self.forces[key]
        self.stats["matched_rows"]+=len(matched)
        self.stats["matched_qps"]+=bool(matched)
        self.stats["matching_seconds"]+=time.perf_counter()-start
        return matched
