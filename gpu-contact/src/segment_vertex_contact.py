"""Finite-overlap vertex inequalities for piecewise-planar contact.

For each visible triangle pair, the frozen signed ray gap is affine on the
overlap polygon. Vertex inequalities bound that gap on the entire polygon.
Multipliers act on ray gaps; physical feature forces include the recorded
normalization scale. They are not mortar pressure coefficients. Coincident
identical virtual-work rows are merged; incompatible normals are not merged.
This diagnostic alternative may lock and requires independent accuracy checks.
It adds no clearance, spring, volume nodes, continuous collision guarantee,
edge-only contact, smooth-surface geometry or scene-specific backend.
"""
import numpy as np
import scipy.sparse as sp
from rtree import index
from mortar_contact_geometry import overlap_quadrature
from segment_contact_kernel import compiled_overlap_vertices,packed_overlap_batch


def feature_search_direction(ns,nm,secondary,main,ray,fallback_cosine=None):
    """Normal to the touching vertex/face or edge/edge feature pair.

    At closure, this eliminates tangential shape-coordinate variations in
    virtual work. A ray direction alone is not the derivative at a tilted
    vertex/face contact. Return the derivative normalized to the ray gap.
    """
    si=np.flatnonzero(ns>1e-8);mi=np.flatnonzero(nm>1e-8)
    if len(si)==1:
        n=np.cross(main[1]-main[0],main[2]-main[0])
    elif len(mi)==1:
        n=ray.copy()
    elif len(si)==2 and len(mi)==2:
        n=np.cross(secondary[si[1]]-secondary[si[0]],main[mi[1]]-main[mi[0]])
    else:raise RuntimeError('Overlap vertex has no vertex/face or edge/edge feature')
    norm=np.linalg.norm(n)
    if norm<1e-24:
        if fallback_cosine is not None:return ray.copy(),1.,True
        raise RuntimeError('Degenerate contact feature normal')
    n=n/norm
    if n@ray<0:n=-n
    cosine=float(n@ray)
    if fallback_cosine is not None and cosine<fallback_cosine:return ray.copy(),1.,True
    if cosine<1e-8:raise RuntimeError(f'Contact feature is singular along the selected ray: cosine={cosine}, ns={ns.tolist()}, nm={nm.tolist()}, secondary={secondary.tolist()}, main={main.tolist()}, ray={ray.tolist()}')
    return n/cosine,1./cosine,False


def contact_feature_normal(ns,nm,secondary,main,ray):
    """Normal to the touching vertex/face or edge/edge feature pair.

    At closure, this eliminates tangential shape-coordinate variations in
    virtual work. A ray direction alone is not the derivative at a tilted
    vertex/face contact. Return the derivative normalized to the ray gap.
    """
    si=np.flatnonzero(ns>1e-8);mi=np.flatnonzero(nm>1e-8)
    if len(si)==1:
        n=np.cross(main[1]-main[0],main[2]-main[0])
    elif len(mi)==1:
        n=ray.copy()
    elif len(si)==2 and len(mi)==2:
        n=np.cross(secondary[si[1]]-secondary[si[0]],main[mi[1]]-main[mi[0]])
    else:raise RuntimeError('Overlap vertex has no vertex/face or edge/edge feature')
    norm=np.linalg.norm(n)
    if norm<1e-24:raise RuntimeError('Degenerate contact feature normal')
    n=n/norm
    if n@ray<0:n=-n
    cosine=float(n@ray)
    if cosine<1e-8:raise RuntimeError(f'Contact feature is singular along the selected ray: cosine={cosine}, ns={ns.tolist()}, nm={nm.tolist()}, secondary={secondary.tolist()}, main={main.tolist()}, ray={ray.tolist()}')
    return n/cosine,1./cosine


class SegmentVertexContact:
    def __init__(self,bodies,offsets,pairs,search_radius,normal_policy='feature',geometry_kernel='compiled',candidate_cache=None,open_gap_tolerance=None,open_feature_cosine_floor=None):
        self.bodies=bodies;self.offsets=offsets;self.pairs=pairs;self.radius=search_radius
        if normal_policy not in ['feature','ray']:raise ValueError(normal_policy)
        self.normal_policy=normal_policy
        if geometry_kernel not in ['compiled','python','batched']:raise ValueError(geometry_kernel)
        self.geometry_kernel=geometry_kernel
        self.candidate_cache=candidate_cache
        self.open_gap_tolerance=open_gap_tolerance
        if open_feature_cosine_floor is not None and not 1e-8<=open_feature_cosine_floor<1.:raise ValueError('Invalid open feature cosine floor')
        self.open_feature_cosine_floor=open_feature_cosine_floor
        self.statistics={'calls':0,'candidate_triangle_pairs':0,'overlap_segments':0,
                         'raw_vertices':0,'merged_rows':0,'maximum_coverage_fraction':0.,'normal_policy':normal_policy,'geometry_kernel':geometry_kernel,'maximum_reaction_scale':1.,'open_search_cut_vertices':0,'open_direction_rows':0}

    def build(self,x,J):
        x=np.asarray(x)
        self.statistics['calls']+=1;assembled={}
        for slave,master in self.pairs:
            fs=self.bodies[slave].spec['contact_faces'];fm=self.bodies[master].spec['contact_faces']
            if not len(fs) or not len(fm):continue
            sm=x[self.offsets[slave]:self.offsets[slave+1]];mm=x[self.offsets[master]:self.offsets[master+1]]
            triangles=mm[fm];lower=triangles.min(axis=1);upper=triangles.max(axis=1)
            cached=self.candidate_cache.query((slave,master),sm,fs,mm,fm,self.radius) if self.candidate_cache else None
            tree=None
            if cached is None:
                properties=index.Property();properties.dimension=3
                tree=index.Index(((i,tuple(np.concatenate((lo,hi))),None) for i,(lo,hi) in enumerate(zip(lower,upper))),properties=properties)
            for fi,face in enumerate(fs):
                if cached is not None:candidates=cached[fi]
                else:
                    secondary=sm[face];box=np.concatenate((secondary.min(axis=0)-self.radius,secondary.max(axis=0)+self.radius))
                    candidates=sorted(tree.intersection(tuple(box)))
                if not len(candidates):continue
                secondary=sm[face];coverage=0.;area=0.
                self.statistics['candidate_triangle_pairs']+=len(candidates)
                if self.geometry_kernel=='batched':
                    packed,segments,raw,cut,opened,scale,fraction=packed_overlap_batch(secondary,triangles,np.asarray(candidates,dtype=np.int64),self.radius,face,fm,
                        self.offsets[slave],self.offsets[master],np.inf if self.open_gap_tolerance is None else self.open_gap_tolerance,self.normal_policy=='feature',
                        -1. if self.open_feature_cosine_floor is None else self.open_feature_cosine_floor)
                    self.statistics['overlap_segments']+=segments;self.statistics['raw_vertices']+=raw
                    self.statistics['open_search_cut_vertices']+=cut;self.statistics['open_direction_rows']+=opened
                    self.statistics['maximum_reaction_scale']=max(self.statistics['maximum_reaction_scale'],scale)
                    self.statistics['maximum_coverage_fraction']=max(self.statistics['maximum_coverage_fraction'],fraction)
                    for m,ids,vec,rounded,normal,reaction_scale,point,weight,gap,open_direction,node in packed:
                        key=(slave,master,tuple(ids.tolist()),tuple(rounded.tolist()))
                        if key in assembled:
                            assembled[key]['area']+=weight;self.statistics['merged_rows']+=1;continue
                        assembled[key]={'ids':ids,'vec':vec,'gap':float(gap),'normal':normal,'reaction_scale':reaction_scale,'point':point,
                            'area':float(weight),'slave':slave,'master':master,'secondary_face':fi,'master_face':m,'open_direction_only':open_direction,'representative_node':node}
                    continue
                else:
                    kernel=compiled_overlap_vertices if self.geometry_kernel=='compiled' else lambda a,b,r:overlap_quadrature(a,b,r,'vertices')
                    rules=((int(m),kernel(secondary,triangles[m],self.radius)) for m in candidates)
                for m,rule in rules:
                    m=int(m)
                    if rule is None:continue
                    Ns,Nm,w,g,n,area=rule;coverage+=w.sum();self.statistics['overlap_segments']+=1
                    self.statistics['raw_vertices']+=len(w)
                    ids0=np.concatenate((face+self.offsets[slave],fm[m]+self.offsets[master]))
                    for k in range(len(w)):
                        # Radius clipping creates artificial polygon corners.
                        # A +radius corner is strictly open, not a physical
                        # vertex/face or edge/edge feature. The affine minimum
                        # on the clipped polygon remains at a retained corner
                        # whenever it is nonpositive. A negative cut means
                        # penetration reached the search horizon: fail, rather
                        # than invent a normal or accept a truncated minimum.
                        if abs(g[k]-self.radius)<=max(1e-13,self.radius*1e-8):
                            self.statistics['open_search_cut_vertices']+=1
                            continue
                        if abs(g[k]+self.radius)<=max(1e-13,self.radius*1e-8):
                            raise RuntimeError('Penetration reached the contact search horizon')
                        # The closure feature derivative contains 1/(n.ray)
                        # and is not a valid far-open closest-feature derivative.
                        # Opt-in stabilized SEARCH directions use the frozen ray
                        # only while strictly open. Such rows cannot carry an
                        # accepted multiplier. At closure retain the real feature.
                        open_direction=self.open_gap_tolerance is not None and g[k]>self.open_gap_tolerance
                        if self.open_feature_cosine_floor is None:
                            normal,reaction_scale=contact_feature_normal(Ns[k],Nm[k],secondary,triangles[m],n) if self.normal_policy=='feature' and not open_direction else (n,1.)
                        elif self.normal_policy=='feature':
                            normal,reaction_scale,open_direction=feature_search_direction(Ns[k],Nm[k],secondary,triangles[m],n,
                                self.open_feature_cosine_floor if open_direction else None)
                        else:normal,reaction_scale=n,1.
                        if open_direction:self.statistics['open_direction_rows']+=1
                        self.statistics['maximum_reaction_scale']=max(self.statistics['maximum_reaction_scale'],reaction_scale)
                        shape=np.concatenate((Ns[k],-Nm[k]));keep=np.abs(shape)>1e-13;ids=ids0[keep];vec=shape[keep,None]*normal
                        order=np.argsort(ids);ids=ids[order];vec=vec[order]
                        key=(slave,master,tuple(ids.tolist()),tuple(np.round(vec.ravel(),11)))
                        if key in assembled:
                            assembled[key]['area']+=w[k];self.statistics['merged_rows']+=1
                            continue
                        assembled[key]={'ids':ids,'vec':vec,'gap':float(g[k]),'normal':normal,'reaction_scale':reaction_scale,'point':Nm[k]@triangles[m],
                                        'area':float(w[k]),'slave':slave,'master':master,'secondary_face':fi,'master_face':m,'open_direction_only':open_direction,
                                        'representative_node':int(face[np.argmax(Ns[k])]+self.offsets[slave])}
                if area:
                    fraction=coverage/area;self.statistics['maximum_coverage_fraction']=max(self.statistics['maximum_coverage_fraction'],float(fraction))
                    if fraction>1.+1e-7:raise RuntimeError('Multiple main sheets require a visibility partition')
            if tree is not None:tree.close()
        rows=[];cols=[];values=[];gaps=[];records=[]
        for row,r in enumerate(assembled.values()):
            ids=r['ids'];rows.extend([row]*3*len(ids));cols.extend((3*ids[:,None]+np.arange(3)).ravel());values.extend(r['vec'].ravel())
            gaps.append(r['gap']);records.append({'key':(r['slave'],r['master'],r['secondary_face'],r['master_face'],row),
                 'ids':np.concatenate(([r['representative_node']],ids)),'normal':r['normal'],'master_face':r['master_face'],'point':r['point'],
                 'support_area':r['area'],'min_sample_gap':r['gap'],'max_sample_gap':r['gap'],'reaction_scale':r['reaction_scale'],'open_direction_only':r['open_direction_only']})
        Gx=sp.csr_matrix((values,(rows,cols)),shape=(len(gaps),x.size))
        return np.asarray(gaps),Gx,(Gx@J).tocsr(),records
