"""Selected crease-median / finite-compliance law on CPU and CUDA.

The contact law is shared with objective_crease_compliant_fast. Structural
models remain explicitly named: this module uses surface CMS, not the
full-coordinate objective-material diagnostic. No body-volume online loop.
"""
from pathlib import Path
from types import SimpleNamespace
import numpy as np

CONTRACT = dict(pressure='secondary_crease_median', crease_angle_degrees=30.,
    gap='area_average', enforcement='element_normal_compliance', compliance_scale=1000.,
    force_time='physical_endpoint', friction='none', alpha=-.05,
    structural_model='surface_cms_generalized_hht')
CARTESIAN_CONTRACT={**CONTRACT,'structural_model':'surface_cms_cartesian_projected_hht',
    'material_model':'linear_body_frame','initial_acceleration':'projected_zero_cartesian'}
ANCHORED_P1_CONTRACT={**CARTESIAN_CONTRACT,'pressure':'nearest_primary_crease_P1'}
P1_CONTRACT={**CARTESIAN_CONTRACT,'pressure':'secondary_crease_P1'}


def create_selected_cpu(scene,bodies,policy=None,sources=None):
    from residual_cms import ResidualSurfaceCMS
    from sector_pressure_contact import CreaseNormalPressureContact
    from finite_contact_compliance import ElementNormalCompliance
    if sources is not None:
        import copy
        from surface_response_source import ResidualColumnView
        bodies=[copy.copy(b) for b in bodies]
        for b,source in zip(bodies,sources): b.R=ResidualColumnView(source)
    cls=ResidualSurfaceCMS
    requested=scene.get('common_contact_contract',CONTRACT)
    if requested in (CARTESIAN_CONTRACT,ANCHORED_P1_CONTRACT,P1_CONTRACT):
        from projected_cartesian_inertia import cpu_class
        cls=cpu_class()
    sim=cls(scene,bodies,policy);old=sim.geometry
    geometry=CreaseNormalPressureContact
    if requested==ANCHORED_P1_CONTRACT:
        from normal_anchor_contact import AnchoredP1PressureContact
        geometry=AnchoredP1PressureContact
    elif requested==P1_CONTRACT:
        from pressure_support_diagnostics import CreaseP1PressureContact
        geometry=CreaseP1PressureContact
    sim.geometry=geometry(bodies,sim.x_offsets,scene['contact_pairs'],old.radius,
        candidate_cache=old.candidate_cache,gap_rule='p1_average',threads=1)
    sim.normal_compliance=ElementNormalCompliance(scene['bodies'],scale=CONTRACT['compliance_scale'])
    sim.contact_contract=scene.get('common_contact_contract',CONTRACT).copy()
    return sim


def gpu_types(anchored_p1=False,continuous_p1=False,sealed_region=False,separated_response=False,streamed_region=False,mesh_factor=False,direct_function=False):
    """Delay importing CUDA so CPU reference processes do not own a context."""
    import cupy as cp
    from device_cached_normal import DeviceCachedNormal
    from cached_normal_contact import FILTER,EXTRA
    from sector_pressure_contact import smooth_components
    from finite_contact_compliance import ElementNormalCompliance
    from windowed_surface_backend import WindowedTimeSurfaceCMS,WindowWorkingSet
    if sealed_region:
        from sealed_region_backend import SealedRegionSurfaceCMS,CompactWorkingSet
        WindowedTimeSurfaceCMS,WindowWorkingSet=SealedRegionSurfaceCMS,CompactWorkingSet
    if separated_response:
        from separated_region_backend import SeparatedRegionSurfaceCMS,SeparatedWorkingSet
        WindowedTimeSurfaceCMS,WindowWorkingSet=SeparatedRegionSurfaceCMS,SeparatedWorkingSet
    if streamed_region:
        from streamed_region_backend import StreamedRegionSurfaceCMS
        from separated_region_backend import SeparatedWorkingSet
        WindowedTimeSurfaceCMS,WindowWorkingSet=StreamedRegionSurfaceCMS,SeparatedWorkingSet
    if mesh_factor:
        from mesh_factor_backend import MeshFactorSurfaceCMS,MeshFactorWorking
        WindowedTimeSurfaceCMS,WindowWorkingSet=MeshFactorSurfaceCMS,MeshFactorWorking
    if direct_function:
        from direct_function_backend import DirectFunctionSurfaceCMS,DirectFunctionWorking
        WindowedTimeSurfaceCMS,WindowWorkingSet=DirectFunctionSurfaceCMS,DirectFunctionWorking
    use_p1=anchored_p1 or continuous_p1

    class SelectedGeometry(DeviceCachedNormal):
        def __init__(self,bodies,scene):
            super().__init__(bodies,scene)
            self.live_support_enabled=bool(scene.get('regional_live_support',False))
            self.live_surface_prepare=bool(scene.get('regional_live_surface_prepare',False))
            self.live_sort_cache=bool(scene.get('regional_live_sort_cache',False))
            self.feature_faces=[];self.feature_keys=[];self.group_counts=[];self.primary_features=[]
            for b in bodies:
                fs=np.asarray(b.spec['contact_faces']);labels=smooth_components(b.X,fs)
                count=int(labels.max()+1);keys,inverse=np.unique(fs*count+labels[:,None],return_inverse=True)
                self.feature_faces.append(cp.asarray(inverse.reshape(fs.shape),dtype=cp.int32))
                self.feature_keys.append(keys);self.group_counts.append(count)
                self.primary_features.append(cp.asarray(labels if anchored_p1 else np.zeros_like(labels),dtype=cp.int32))
            self.maxgroups=max(len(k) for k in self.feature_keys)+1
            law=ElementNormalCompliance(scene['bodies'],scale=CONTRACT['compliance_scale'])
            self.face_stiffness=[cp.asarray(k) for k in law.face_k]
            src=Path(__file__).with_name('normal_contact.cu').read_text()
            src=src.replace('double*out,int n){','double*out,int n,const int*features){',1)
            src=src.replace('out+3*faces[3*i+j]+k','out+3*features[3*i+j]+k')
            src=src.replace('double*piece_values,int*piece_nodes,int*piece_row,double*minimum,int*error){',
                'double*piece_values,int*piece_nodes,int*piece_row,double*minimum,int*error,const int*features,const double*face_k,double*stiffness){')
            src=src.replace('V<double,3>s[3],m[3];','if(face<0||master<0)return;\n V<double,3>s[3],m[3];',1)
            src=src.replace('auto outward=read_point(normals,fs[3*face+vertex]);',
                FILTER+'auto outward=read_point(normals,features[3*face+vertex]);')
            src=src.replace('int row=fs[3*face+vertex];',
                'int row=features[3*face+vertex];atomicAdd(stiffness+row,a*scale*face_k[face]);')
            if use_p1:
                from anchored_pressure_kernels import transform
                src=transform(src)
            topology=bool(scene.get('regional_clip_topology_cache',False))
            if topology and use_p1:raise NotImplementedError('Topology replay is qualified for the median-pressure contract')
            overlap=Path(__file__).with_name('overlap.cu').read_text()
            if scene.get('regional_integral_moments',False):
                if topology:raise ValueError('Moment integration and clipping replay must be qualified separately')
                from polygon_moment_integrals import specialize
                overlap=specialize(overlap)
            if topology:
                from clipping_topology_cache import overlap_source,patch_source
                overlap=overlap_source(overlap);src=patch_source(src)
            if scene.get('regional_guard_pressure_reuse',False):
                marker='ctx.slot=5;auto patch=evaluate(f,embedded,m,band,false,radius,&ctx);' if topology else 'auto patch=evaluate(f,embedded,m,band,false,radius);'
                replacement='bool same_polygon=true; for(int j=0;j<guard.count;j++)same_polygon=same_polygon&&guard.vg[j]>=-radius&&guard.vg[j]<=fmin(radius,band); '
                replacement+=('ctx.slot=5;auto patch=same_polygon?guard:evaluate(f,embedded,m,band,false,radius,&ctx);' if topology else 'auto patch=same_polygon?guard:evaluate(f,embedded,m,band,false,radius);')
                assert src.count(marker)==1;src=src.replace(marker,replacement)
            self.topology_cache_enabled=topology
            module=cp.RawModule(code=overlap+src+EXTRA)
            self.original_k={n:module.get_function(n) for n in ('nodal_normals','normal_patches','check_normal_coverage','pack_normal_rows','cache_valid','fixed_rows')}
            self.k=dict(self.original_k)
            if use_p1:
                self.choose=module.get_function('choose_feature');self.accumulate=module.get_function('accumulate_feature')
            def normals(grid,block,args):
                i=next(i for i,f in enumerate(self.faces) if f.data.ptr==args[1].data.ptr)
                self.original_k['nodal_normals'](grid,block,tuple(args)+(self.feature_faces[i],))
            def patches(grid,block,args):
                p=next(p for p in self.parts if p.owners.data.ptr==args[4].data.ptr)
                p.stiffness.fill(0)
                extra=(self.feature_faces[p.slave],self.face_stiffness[p.slave],p.stiffness)
                if use_p1:
                    p.best.fill(cp.inf);p.chosen.fill(np.iinfo(np.int32).max)
                    extra+=(self.primary_features[p.master],p.piece_integrals,p.piece_distances,p.piece_features,p.best)
                if topology:extra+=(p.clip_cache,p.clip_hits,p.clip_misses)
                self.original_k['normal_patches'](grid,block,tuple(args)+extra)
                if use_p1:
                    n=np.int32(len(p.rows))
                    self.choose(grid,block,(p.rows,p.piece_distances,p.piece_features,p.best,p.chosen,n))
                    self.accumulate(grid,block,(p.rows,p.piece_features,p.chosen,p.piece_integrals,p.area,p.integral,p.stiffness,p.counts,n))
            self.k['nodal_normals']=normals;self.k['normal_patches']=patches
            self.statistics.update(online_geometry='cuda_crease_median_compliant',contact_contract=CONTRACT.copy())

        def refresh(self,x):
            changed=super().refresh(x)
            if not changed and hasattr(self,'compliance'):return False
            nr=0;keys=[]
            for p in self.parts:
                n=len(self.feature_keys[p.slave]);p.offset=nr
                p.nodes=cp.arange(n,dtype=cp.int32);p.lut=p.nodes+nr
                p.area=cp.zeros(n);p.integral=cp.zeros(n);p.counts=cp.zeros(n,dtype=cp.int32);p.stiffness=cp.zeros(n)
                if self.topology_cache_enabled:
                    p.clip_cache=cp.zeros((len(p.rows),14),dtype=cp.uint64)
                    p.clip_hits=cp.zeros(len(p.rows),dtype=cp.uint64);p.clip_misses=cp.zeros(len(p.rows),dtype=cp.uint64)
                if use_p1:
                    p.best=cp.empty(n);p.chosen=cp.empty(n,dtype=cp.int32)
                    p.piece_integrals=cp.empty((len(p.rows),3));p.piece_distances=cp.empty(len(p.rows));p.piece_features=cp.empty(len(p.rows),dtype=cp.int32)
                keys.append(((p.slave*len(self.bodies)+p.master)*self.maxgroups+p.nodes).astype(cp.int64));nr+=n
            self.normals={i:cp.zeros((len(self.feature_keys[i]),3)) for i in sorted(set(p[0] for p in self.pairs))}
            self.gaps=cp.zeros(nr);self.valid=cp.zeros(nr,dtype=cp.int32);self.counts=cp.zeros(nr,dtype=cp.int32)
            self.ptr=cp.zeros(nr+1,dtype=cp.int32);self.cursor=cp.zeros(nr,dtype=cp.int32)
            self.keys=cp.concatenate(keys);self.compliance=cp.zeros(nr)
            self.base_keys=self.keys.copy()
            self.G=SimpleNamespace(indptr=self.ptr,indices=self.cols,data=self.values,shape=(nr,self.size))
            from device_geometry import DeviceWorld
            self.world=DeviceWorld(self.gaps,self.G,self.keys,self.minimum.reshape(()),self.statistics)
            return True

        def enqueue(self,x):
            world=super().enqueue(x)
            for p in self.parts:
                n=len(p.nodes)
                self.compliance[p.offset:p.offset+n]=cp.where(p.stiffness>0,1/cp.maximum(p.stiffness,1e-300),0.)
                if anchored_p1:
                    sl=slice(p.offset,p.offset+n)
                    self.keys[sl]=self.base_keys[sl]*(max(self.group_counts)+1)+cp.where(p.chosen<np.iinfo(np.int32).max,p.chosen+1,0)
            return world

    extra=r'''
extern "C" __global__ void add_compliance(double*W,const double*c,const long long*ids,const int*meta,int cap){
 int j=blockIdx.x*blockDim.x+threadIdx.x;
 if(meta[4]||meta[5]||j<meta[3]||j>=meta[2])return;
 W[ids[j]*cap+j]+=c[ids[j]];
}
'''
    class SelectedWorking(WindowWorkingSet):
        def __init__(self,device,*args,**kw):
            super().__init__(device,*args,**kw);kernel=cp.RawKernel(extra,'add_compliance')
            original=self.region_k['region_contact']
            def contact(grid,block,args):
                original(grid,block,args)
                if args[3].data.ptr==self.buffers[-1][3].data.ptr:
                    kernel(((self.cap+127)//128,),(128,),
                        (self.W,device.geometry.compliance,self.ids,self.meta,np.int32(self.cap)))
            self.region_k['region_contact']=contact

    if streamed_region:
        from streamed_region_backend import regional_geometry
        SelectedGeometry=regional_geometry(SelectedGeometry)

    class SelectedSurfaceCMS(WindowedTimeSurfaceCMS):
        def rebuild(self,refresh=True):
            if not isinstance(self.geometry,SelectedGeometry):self.geometry=SelectedGeometry(self.bodies,self.scene)
            if not isinstance(self.graph_working,SelectedWorking):
                cap=int(self.scene.get('regional_initial_working_capacity',32))
                if cap not in (32,64,128,256):raise ValueError('Initial working capacity must be 32, 64, 128 or 256')
                self.graph_working=SelectedWorking(self,cap=cap)
            return super().rebuild(refresh)

        def after_contact(self):
            # finish is captured before the effective-gap calculation; keep raw
            # geometry unchanged for output, subsequent assembly and BVH checks.
            original_finish=self.graph_working.enqueue_finish
            raw=self.world.gaps
            def finish(keys):
                lam=original_finish(keys)
                self.world.gaps=raw+self.geometry.compliance*lam
                return lam
            self.graph_working.enqueue_finish=finish
            try:return super().after_contact()
            finally:
                self.graph_working.enqueue_finish=original_finish;self.world.gaps=raw

    return SelectedGeometry,SelectedSurfaceCMS


def create_selected_gpu(scene,bodies,policy=None,batch_capacity=64,sources=None,sealed_region=False,separated_response=False,streamed_region=False,mesh_factor=False,direct_function=False):
    # This is a storage budget, not a change to contact tolerances or equations.
    # The resolved helical model needs 276.5 MiB for its initial candidate set.
    budget=int(scene.get('response_window_bytes',512*2**20))
    import cupy as cp
    _,total=cp.cuda.runtime.memGetInfo()
    if not 0<budget<=total-1024*2**20:
        raise ValueError('Response budget must leave at least 1 GiB of device capacity for non-cache work')
    requested=scene.get('common_contact_contract',CONTRACT)
    _,cls=gpu_types(requested==ANCHORED_P1_CONTRACT,requested==P1_CONTRACT,sealed_region,separated_response,streamed_region,mesh_factor,direct_function)
    if requested in (CARTESIAN_CONTRACT,ANCHORED_P1_CONTRACT,P1_CONTRACT):
        from projected_cartesian_inertia import gpu_class
        cls=gpu_class(cls)
    sim=cls(scene,bodies,policy,batch_capacity,sources=sources,window_bytes=budget)
    sim.contact_contract=scene.get('common_contact_contract',CONTRACT).copy()
    return sim
