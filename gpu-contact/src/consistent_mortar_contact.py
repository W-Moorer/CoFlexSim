"""Shared finite-overlap weak contact with the total derivative of each gap.

The nonnegative normalized P1 average uses the current finite overlap. The
adjoint contact force uses its exact piecewise-smooth derivative, including
normal, shape, clipping, area and normalization terms. This is a declared weak
FE discretization, not a reverse-engineered Abaqus implementation. Positive
and negative point gaps within one active support must be reported separately.
"""
import sys,time
import numpy as np
import scipy.sparse as sp
from rtree import index
from coflex_core import BUILD
from contact_candidate_cache import ContactCandidateCache,CandidateRows
sys.path.insert(0,str(BUILD/'mortar_native/Release'))
import coflex_mortar_native


class ContactGeometryError(RuntimeError):
    """Undefined multi-sheet weak support; trial states may be backtracked."""


class ConsistentMortarContact:
    normal_policy='total_weak_gap_derivative'
    open_gap_tolerance=None
    open_feature_cosine_floor=None
    geometry_kernel='autodiff'

    @classmethod
    def for_scene(cls,bodies,offsets,scene,radius,**options):
        pairs=scene.get('contact_pairs',[])
        if scene.get('contact_patch_bound_m') is not None:
            if scene.get('contact_skins') or scene.get('contact_face_blend_m') is not None or options.get('gap_rule')!='overlap_vertices':
                raise ValueError('Bounded local mortar requires the original finite-overlap vertex guard')
            from bounded_patch_contact import BoundedPatchContact
            geometry=BoundedPatchContact
            if scene.get("shared_working_contact"):
                from shared_contact import SharedBoundedContact
                geometry=SharedBoundedContact
            elif scene.get('contact_pressure_support','legacy_symmetric')!='legacy_symmetric':
                raise ValueError('Upper-bounded pressure support requires the shared overlap backend')
            return geometry(bodies,offsets,pairs,radius,bound=scene['contact_patch_bound_m'],
                                       reserve=scene.get("contact_guard_reserve_m",0.),pressure_support=scene.get('contact_pressure_support','legacy_symmetric'),
                                       pressure_band=scene.get('contact_pressure_band_m'),**options)
        if scene.get('contact_face_blend_m') is not None:
            if scene.get('contact_skins') or options.get('gap_rule')!='overlap_vertices':
                raise ValueError('Face blending uses the original finite-overlap vertex gaps')
            from blended_face_contact import BlendedFaceContact
            return BlendedFaceContact(bodies,offsets,pairs,radius,blend=scene['contact_face_blend_m'],**options)
        if scene.get('contact_skins'):
            from embedded_contact_skin import EmbeddedMortarContact
            return EmbeddedMortarContact(bodies,offsets,pairs,radius,skins=scene['contact_skins'],**options)
        return cls(bodies,offsets,pairs,radius,**options)

    def __init__(self,bodies,offsets,pairs,search_radius,candidate_cache=None,threads=1,gap_rule='p1_average'):
        self.bodies=bodies;self.offsets=offsets;self.pairs=pairs;self.radius=search_radius
        self.candidate_cache=candidate_cache or ContactCandidateCache(.25*search_radius,16*1024**2)
        if int(threads)!=threads or threads<1:raise ValueError('Positive integer contact thread count required')
        self.threads=int(threads)
        if gap_rule not in ('p1_average','overlap_minimum','overlap_vertices'):raise ValueError('Unknown finite-overlap gap rule')
        self.gap_rule=gap_rule
        self.statistics=dict(calls=0,candidate_triangle_pairs=0,overlap_segments=0,quadrature_points=0,
            maximum_coverage_fraction=0.,geometry_id=coflex_mortar_native.geometry_id,
            gap_kind=gap_rule,force_kind='adjoint_of_total_gap_derivative',threads=self.threads,
            timings_seconds={k:0. for k in ['candidate_query','pair_packing','native_allocation','scalar_geometry','geometry_derivatives','native_bridge','support_assembly','derivative_assembly','record_assembly','remaining_geometry']},total_build_seconds=0.)

    def build(self,x,J,derivatives=True,gaps_only=False):
        tick=time.perf_counter();timing=self.statistics['timings_seconds'];before=sum(timing.values())
        if gaps_only and derivatives:raise ValueError('Gap-only query cannot request derivatives')
        try:return self._build(x,J,derivatives,gaps_only)
        finally:
            elapsed=time.perf_counter()-tick;timing['remaining_geometry']+=elapsed-(sum(timing.values())-before)
            self.statistics['total_build_seconds']+=elapsed

    def build_gaps(self,x):
        """Identical candidate/overlap checks without unused trial records."""
        return self.build(x,None,derivatives=False,gaps_only=True)

    def pair_inputs(self,x):
        timing=self.statistics["timings_seconds"]
        for slave,master in sorted(self.pairs):
            fs=self.bodies[slave].spec['contact_faces'];fm=self.bodies[master].spec['contact_faces']
            if not len(fs) or not len(fm):continue
            sm=x[self.offsets[slave]:self.offsets[slave+1]];mm=x[self.offsets[master]:self.offsets[master+1]]
            tick=time.perf_counter()
            candidates=self.candidate_cache.query((slave,master),sm,fs,mm,fm,self.radius)
            if candidates is None:
                tri=mm[fm];prop=index.Property();prop.dimension=3
                tree=index.Index(((i,tuple(np.r_[t.min(0),t.max(0)]),None) for i,t in enumerate(tri)),properties=prop)
                try:candidates=[np.array(sorted(tree.intersection(tuple(np.r_[sm[f].min(0)-self.radius,sm[f].max(0)+self.radius]))),int) for f in fs]
                finally:tree.close()
            timing['candidate_query']+=time.perf_counter()-tick;tick=time.perf_counter()
            if isinstance(candidates,CandidateRows):
                owners=np.repeat(np.arange(len(fs)),np.diff(candidates.offsets));mids=candidates.indices
            else:owners=np.repeat(np.arange(len(fs)),[len(c) for c in candidates]);mids=np.concatenate(candidates).astype(int)
            if not len(owners):timing['pair_packing']+=time.perf_counter()-tick;continue
            self.statistics['candidate_triangle_pairs']+=len(owners)
            secondary_tri=sm[fs[owners]];main_tri=mm[fm[mids]]
            if hasattr(self,"screen_separated_pairs"):
                owners,mids,secondary_tri,main_tri=self.screen_separated_pairs(master,owners,mids,secondary_tri,main_tri)
                if not len(owners):timing['pair_packing']+=time.perf_counter()-tick;continue
            timing["pair_packing"]+=time.perf_counter()-tick
            yield slave,master,fs,fm,owners,mids,secondary_tri,main_tri

    def _build(self,x,J,derivatives=True,gaps_only=False):
        """Sparse batch assembly of the identical integrated weak constraint."""
        timing=self.statistics['timings_seconds']
        self.statistics['calls']+=1;x=np.asarray(x);all_gaps=[];all_records=[];all_matrices=[]
        for slave,master,fs,fm,owners,mids,secondary_tri,main_tri in self.pair_inputs(x):
            tick=time.perf_counter()
            secondary_tri=secondary_tri.reshape(-1,9);main_tri=main_tri.reshape(-1,9)
            timing['pair_packing']+=time.perf_counter()-tick;tick=time.perf_counter()
            result=coflex_mortar_native.batch(secondary_tri,main_tri,self.radius,derivatives,self.threads,self.gap_rule!='p1_average',getattr(self,'profile_overlap',False))
            native_elapsed=time.perf_counter()-tick
            if getattr(self,'profile_overlap',False):
                for key in ('scalar_overlap_s','ad_overlap_s','scalar_clip_s','ad_clip_s'):
                    if key not in result:continue
                    self.statistics[key]=self.statistics.get(key,0.)+result[key]
            for k,v in result['timings_seconds'].items():timing[k]+=v
            timing['native_bridge']+=native_elapsed-sum(result['timings_seconds'].values())
            tick=time.perf_counter()
            selected=np.flatnonzero(result['samples']);owners=owners[selected];mids=mids[selected]
            if not len(selected):timing['support_assembly']+=time.perf_counter()-tick;continue
            aa=result['area'][selected];info=result['info'][selected]
            coverage=np.bincount(owners,aa.sum(axis=1),minlength=len(fs));reference_area=np.zeros(len(fs));reference_area[owners]=info[:,5]
            fraction=coverage[reference_area>0]/reference_area[reference_area>0]
            self.statistics['maximum_coverage_fraction']=max(self.statistics['maximum_coverage_fraction'],float(fraction.max(initial=0.)))
            if fraction.max(initial=0.)>1.+1e-7:raise ContactGeometryError(f'Multiple main sheets overlap a secondary triangle; visibility partition required (coverage={fraction.max():.16g})')
            self.statistics['overlap_segments']+=len(selected);self.statistics['quadrature_points']+=int(result['samples'][selected].sum())
            if self.gap_rule!='p1_average':
                if self.gap_rule=='overlap_vertices':
                    pairrow,vertex=np.nonzero(np.arange(12)[None,:]<result['vertex_counts'][selected,None])
                    if result['vertex_counts'][selected].max(initial=0)>12:raise ContactGeometryError('Overlap polygon exceeds vertex capacity')
                    take=selected[pairrow];gaps=result['vertex_gaps'][take,vertex]
                    derivatives_array=result['vertex_derivatives'][take].reshape(-1,12,18)[np.arange(len(take)),vertex] if derivatives else None
                    points=result['vertex_points'][take].reshape(-1,12,3)[np.arange(len(take)),vertex]
                    ids=np.c_[fs[owners[pairrow]]+self.offsets[slave],fm[mids[pairrow]]+self.offsets[master]]
                else:
                    pairrow=np.arange(len(selected));vertex=np.zeros(len(selected),int);take=selected
                    gaps=result['vertex_gap'][selected];derivatives_array=result['vertex_derivative'][selected] if derivatives else None
                    points=result['vertex_point'][selected]
                    ids=np.c_[fs[owners]+self.offsets[slave],fm[mids]+self.offsets[master]]
                if np.any(gaps < -self.radius):raise ContactGeometryError('Physical overlap extends beyond the negative search boundary')
                blend=None
                if hasattr(self,"face_blending"):
                    blend=self.face_blending(gaps,owners[pairrow])
                if gaps_only:
                    all_gaps.append(gaps if blend is None else blend[0]);timing['support_assembly']+=time.perf_counter()-tick
                    continue
                timing['support_assembly']+=time.perf_counter()-tick;tick=time.perf_counter()
                if derivatives:
                    cols=(3*ids[:,:,None]+np.arange(3)).reshape(-1,18)
                    G=sp.csr_matrix((derivatives_array.ravel(),(np.repeat(np.arange(len(gaps)),18),cols.ravel())),shape=(len(gaps),x.size))
                else:G=sp.csr_matrix((len(gaps),x.size))
                timing['derivative_assembly']+=time.perf_counter()-tick;tick=time.perf_counter()
                if blend is None:
                    records=[dict(key=(slave,master,int(owners[j]),int(mids[j]),int(vertex[k])),ids=ids[k],normal=info[j,:3],master_face=int(mids[j]),
                        point=points[k],support_area=float(aa[j].sum()),min_sample_gap=float(gaps[k]),
                        max_sample_gap=float(info[j,4]),derivative='total_physical_'+self.gap_rule) for k,j in enumerate(pairrow)]
                else:
                    gaps,W,mins,maxs,faces=blend
                    G=(W@G).tocsr()
                    points=W@points
                    largest=np.asarray(W.argmax(axis=1)).ravel()
                    face_area=np.bincount(owners,weights=aa.sum(axis=1),minlength=len(fs))
                    records=[dict(key=(slave,master,int(face)),ids=ids[k],normal=info[pairrow[k],:3],
                        master_face=int(mids[pairrow[k]]),point=points[i],support_area=float(face_area[face]),
                        min_sample_gap=float(mins[i]),max_sample_gap=float(maxs[i]),derivative='simplex_blended_face_gap')
                        for i,(face,k) in enumerate(zip(faces,largest))]
                all_gaps.append(gaps);all_records.extend(records);all_matrices.append(G)
                timing['record_assembly']+=time.perf_counter()-tick
                continue
            nodes,inv=np.unique(fs[owners].ravel(),return_inverse=True);n=len(nodes)
            area=np.bincount(inv,aa.ravel(),minlength=n);weighted=np.bincount(inv,result['weighted_gap'][selected].ravel(),minlength=n)
            gaps=weighted/np.maximum(area,1e-300)
            if gaps_only:
                all_gaps.append(gaps[area>1e-24]);timing['support_assembly']+=time.perf_counter()-tick
                continue
            points=np.column_stack([np.bincount(inv,result['points'][selected].reshape(-1,3)[:,d],minlength=n) for d in range(3)])
            normals=np.column_stack([np.bincount(inv,(aa*info[:,d,None]).ravel(),minlength=n) for d in range(3)])
            minimum=np.full(n,np.inf);maximum=np.full(n,-np.inf)
            np.minimum.at(minimum,inv,np.repeat(info[:,3],3));np.maximum.at(maximum,inv,np.repeat(info[:,4],3))
            support_faces=sp.csr_matrix((aa.ravel(),(inv,np.repeat(mids,3))),shape=(n,len(fm)))
            timing['support_assembly']+=time.perf_counter()-tick;tick=time.perf_counter()
            if derivatives:
                ids=np.c_[fs[owners]+self.offsets[slave],fm[mids]+self.offsets[master]]
                cols=np.repeat((3*ids[:,:,None]+np.arange(3)).reshape(-1,1,18),3,axis=1).ravel()
                vals=(result['gap_derivative'][selected].reshape(-1,18)-gaps[inv,None]*result['area_derivative'][selected].reshape(-1,18))/np.maximum(area[inv,None],1e-300)
                G=sp.csr_matrix((vals.ravel(),(np.repeat(inv,18),cols)),shape=(n,x.size))
                if not np.isfinite(G.data).all():raise RuntimeError('Nonfinite weak-gap derivative')
            else:G=sp.csr_matrix((n,x.size))
            timing['derivative_assembly']+=time.perf_counter()-tick;tick=time.perf_counter()
            keep=np.flatnonzero(area>1e-24);records=[]
            for k in keep:
                begin,end=support_faces.indptr[k:k+2]
                main_face=int(support_faces.indices[begin+np.argmax(support_faces.data[begin:end])])
                touched=np.unique(G.indices[G.indptr[k]:G.indptr[k+1]]//3)
                records.append(dict(key=(slave,master,int(nodes[k])),ids=np.r_[nodes[k]+self.offsets[slave],touched],normal=normals[k]/area[k],
                    master_face=main_face,point=points[k]/area[k],support_area=area[k],min_sample_gap=minimum[k],max_sample_gap=maximum[k],derivative='total_piecewise_smooth'))
            all_gaps.append(gaps[keep]);all_records.extend(records);all_matrices.append(G[keep])
            timing['record_assembly']+=time.perf_counter()-tick
        gaps=np.concatenate(all_gaps) if all_gaps else np.empty(0)
        if gaps_only:return gaps
        Gx=sp.vstack(all_matrices,format='csr') if all_matrices else sp.csr_matrix((0,x.size))
        return gaps,Gx,(Gx@J).tocsr(),all_records

    def build_scalar_assembly_reference(self,x,J,derivatives=True):
        self.statistics['calls']+=1;x=np.asarray(x);assembled={}
        for slave,master in self.pairs:
            fs=self.bodies[slave].spec['contact_faces'];fm=self.bodies[master].spec['contact_faces']
            if not len(fs) or not len(fm):continue
            sm=x[self.offsets[slave]:self.offsets[slave+1]];mm=x[self.offsets[master]:self.offsets[master+1]]
            candidates=self.candidate_cache.query((slave,master),sm,fs,mm,fm,self.radius)
            if candidates is None:
                tri=mm[fm];prop=index.Property();prop.dimension=3
                tree=index.Index(((i,tuple(np.r_[t.min(0),t.max(0)]),None) for i,t in enumerate(tri)),properties=prop)
                try:candidates=[np.array(sorted(tree.intersection(tuple(np.r_[sm[f].min(0)-self.radius,sm[f].max(0)+self.radius]))),int) for f in fs]
                finally:tree.close()
            owners=np.repeat(np.arange(len(fs)),[len(c) for c in candidates]);mids=np.concatenate(candidates).astype(int)
            if not len(owners):continue
            self.statistics['candidate_triangle_pairs']+=len(owners)
            result=coflex_mortar_native.batch(sm[fs[owners]].reshape(-1,9),mm[fm[mids]].reshape(-1,9),self.radius,derivatives)
            coverage=np.zeros(len(fs));reference_area=np.zeros(len(fs))
            for p in np.flatnonzero(result['samples']):
                fi=int(owners[p]);mi=int(mids[p]);face=fs[fi];ids=np.r_[face+self.offsets[slave],fm[mi]+self.offsets[master]]
                aa=result['area'][p];gg=result['weighted_gap'][p];info=result['info'][p]
                if derivatives:da=result['area_derivative'][p].reshape(3,6,3);dg=result['gap_derivative'][p].reshape(3,6,3)
                coverage[fi]+=aa.sum();reference_area[fi]=info[5]
                self.statistics['overlap_segments']+=1;self.statistics['quadrature_points']+=int(result['samples'][p])
                for k,node in enumerate(face):
                    key=(slave,master,int(node));r=assembled.setdefault(key,dict(area=0.,gap=0.,da={},dg={},point=np.zeros(3),normal=np.zeros(3),faces={},min_sample_gap=np.inf,max_sample_gap=-np.inf))
                    r['area']+=aa[k];r['gap']+=gg[k];r['point']+=result['points'][p,3*k:3*k+3];r['normal']+=aa[k]*info[:3]
                    r['min_sample_gap']=min(r['min_sample_gap'],info[3]);r['max_sample_gap']=max(r['max_sample_gap'],info[4]);r['faces'][mi]=r['faces'].get(mi,0.)+aa[k]
                    if derivatives:
                        for j,node2 in enumerate(ids):
                            node2=int(node2);r['da'][node2]=r['da'].get(node2,np.zeros(3))+da[k,j];r['dg'][node2]=r['dg'].get(node2,np.zeros(3))+dg[k,j]
            fraction=coverage[reference_area>0]/reference_area[reference_area>0]
            self.statistics['maximum_coverage_fraction']=max(self.statistics['maximum_coverage_fraction'],float(fraction.max(initial=0.)))
            if fraction.max(initial=0.)>1.+1e-7:raise ContactGeometryError(f'Multiple main sheets overlap a secondary triangle; visibility partition required (coverage={fraction.max():.16g})')
        gaps=[];records=[];rows=[];cols=[];values=[]
        for key,r in sorted(assembled.items()):
            if r['area']<=1e-24:continue
            gap=r['gap']/r['area'];ids=np.array(sorted(r['dg']),int)
            vec=np.array([(r['dg'][j]-gap*r['da'][j])/r['area'] for j in ids])
            if not np.isfinite(vec).all():raise RuntimeError('Nonfinite weak-gap derivative')
            row=len(gaps);gaps.append(gap);rows.extend([row]*3*len(ids));cols.extend((3*ids[:,None]+np.arange(3)).ravel());values.extend(vec.ravel())
            records.append(dict(key=key,ids=np.r_[key[2]+self.offsets[key[0]],ids],normal=r['normal']/r['area'],
                master_face=max(r['faces'],key=r['faces'].get),point=r['point']/r['area'],support_area=r['area'],
                min_sample_gap=r['min_sample_gap'],max_sample_gap=r['max_sample_gap'],derivative='total_piecewise_smooth'))
        Gx=sp.csr_matrix((values,(rows,cols)),shape=(len(gaps),x.size))
        return np.asarray(gaps),Gx,(Gx@J).tocsr(),records
