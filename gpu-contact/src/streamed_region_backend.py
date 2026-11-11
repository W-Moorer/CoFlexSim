"""GPU regional residual blocks; global host library and certified lazy geometry.

All nodes used by a candidate face or its normal's one-ring are restored exactly.
Other nodes keep zero residual displacement for broad phase only. A whole-library
entry bound certifies their omitted displacement; BVH envelopes include that bound.
Any new face outside the prepared one-ring or a bound violation aborts/retries the
unaccepted step. No uncached contact is silently evaluated as rigid.
"""
import time
from pathlib import Path
import numpy as np
import cupy as cp
from device_simulator import sync
from separated_region_backend import SeparatedRegionSurfaceCMS,SeparatedWorkingSet
from sealed_region_backend import SealedContact
from windowed_surface_backend import WindowedTimeSurfaceCMS

CODE=r'''
extern "C" __global__ void regional_recover(const double*R,const double*f,
 const int*force_ids,const int*force_count,const int*lookup,double*out,int n,int cap){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int row=lookup[i];
 if(row<0){out[i]=0.;return;}double v=0.;
 for(int j=0;j<force_count[0];j++){int c=force_ids[j],slot=lookup[c];if(slot>=0)v+=R[(long long)slot*cap+row]*f[c];}
 out[i]=v;
}
extern "C" __global__ void gather_load(const double*f,int*ids,int*count,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&f[i]!=0.)ids[atomicAdd(count,1)]=i;
}
extern "C" __global__ void certify_bound(const double*bound,double allowed,int*error){
 if(!threadIdx.x&&bound[0]>allowed)atomicOr(error,16384);
}
extern "C" __global__ void check_faces(const int*owners,const int*mids,int n,
 const int*slave_ready,const int*master_ready,int*error){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 int a=owners[i],b=mids[i];if(a>=0&&b>=0&&(!slave_ready[a]||!master_ready[b]))atomicOr(error,8192);
}
extern "C" __global__ void mark_ready(const int*ptr,const int*nodes,const int*lookup,int*out,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int ready=1;
 for(int e=ptr[i];e<ptr[i+1]&&ready;e++){int v=3*nodes[e];ready=lookup[v]>=0&&lookup[v+1]>=0&&lookup[v+2]>=0;}
 out[i]=ready;
}
'''


class RegionalRecovery:
    def __init__(self, sim, i):
        self.sim=sim;self.i=i;self.n=len(sim.bodies[i].P)
        self.modal_count=0;self.stored_bytes=0
        self.value=cp.zeros(self.n);self.counts=cp.zeros(2,dtype=cp.int64)
        self.last_bound=cp.zeros(1)
        self.force_ids=cp.empty(self.n,dtype=cp.int32);self.force_count=cp.zeros(1,dtype=cp.int32)

    def apply(self, f, modal_force=None):
        sim=self.sim;B=sim.dynamic[self.i]
        self.force_count.fill(0)
        sim.gather_kernel(((self.n+127)//128,),(128,),(f,self.force_ids,self.force_count,np.int32(self.n)))
        sim.regional_kernel(((self.n+127)//128,),(128,),
            (B.C,f,self.force_ids,self.force_count,B.window_lookup,self.value,np.int32(self.n),np.int32(len(B.window_slots))))
        # Rotation maps componentwise material bounds to a Cartesian infinity bound.
        lo,hi=sim.soff[self.i:self.i+2]
        outside=cp.where(B.window_lookup<0,cp.abs(sim.ts[lo:hi]),0.)
        self.last_bound[:]=np.sqrt(3.)*(cp.sum(B.outside_column_bounds*cp.abs(f))+cp.max(outside))
        sim.certify_kernel((1,),(1,),(self.last_bound,sim.envelope_device if sim.scene.get('regional_native_runtime',False) else np.float64(sim.residual_envelope),sim.geometry.error))
        return self.value


def regional_geometry(base):
    class Geometry(base):
        def __init__(self,bodies,scene):
            super().__init__(bodies,scene)
            self.base_skin=self.skin
            self.pad=float(scene.get('regional_residual_envelope_m',5e-5))
            self.skin=self.base_skin+2*self.pad
            self.query_radius_device.fill(self.radius+self.skin)
            self.region_owner=None
            self.filter_region_pairs=bool(scene.get('regional_current_pair_filter',False))
            if self.filter_region_pairs:
                from regional_pair_filter import prepare
                prepare(self)
            # Reserve 2*pad for omitted displacements; motion consumes base_skin only.
            original=self.dk['need_refresh']
            self.dk['need_refresh']=lambda grid,block,args:original(grid,block,
                tuple(args[:3])+(np.float64(self.base_skin),)+tuple(args[4:]))
            original_valid=self.k['cache_valid']
            self.k['cache_valid']=lambda grid,block,args:original_valid(grid,block,
                tuple(args[:3])+(np.float64(self.base_skin),)+tuple(args[4:]))

        def refresh(self,x):
            self.query_x=x
            return super().refresh(x)

        def enqueue(self,x):
            self.query_x=x
            world=super().enqueue(x)
            if self.region_owner is not None and hasattr(self.region_owner,'face_ready'):
                for p in self.parts:
                    if self.filter_region_pairs:
                        from regional_pair_filter import check
                        check(self,p,x,self.region_owner.face_ready)
                    else:
                        self.region_owner.face_kernel(((len(p.owners)+127)//128,),(128,),
                            (p.owners,p.mids,np.int32(len(p.owners)),self.region_owner.face_ready[p.slave],
                             self.region_owner.face_ready[p.master],self.error))
            return world
    return Geometry


class StreamedRegionSurfaceCMS(SeparatedRegionSurfaceCMS):
    def __init__(self,scene,*args,**kw):
        self.incremental_updates=bool(scene.get('regional_incremental_updates',False))
        self.compiled_region_query=bool(scene.get('regional_compiled_query',False))
        if self.compiled_region_query:
            from packed_surface_query import prepare
            for source in kw.get('sources',[]):prepare(source)
        self.residual_envelope=float(scene.get('regional_residual_envelope_m',5e-5))
        self.envelope_device=cp.asarray([self.residual_envelope],dtype=cp.float64)
        code=CODE
        if scene.get('regional_native_runtime',False):
            code=code.replace('double allowed,int*error){','const double* allowed,int*error){').replace('bound[0]>allowed','bound[0]>allowed[0]')
        module=cp.RawModule(code=code)
        self.regional_kernel=module.get_function('regional_recover')
        self.gather_kernel=module.get_function('gather_load')
        self.certify_kernel=module.get_function('certify_bound')
        self.face_kernel=module.get_function('check_faces')
        self.ready_kernel=module.get_function('mark_ready')
        self.face_ready=[];self.halo=[];self.bound_retries=0
        self.retry_required=[];self.retry_events=[]
        self.region_epoch=0
        self.history_fraction=float(scene.get('regional_history_fraction',.5))
        if not 0<=self.history_fraction<=1:raise ValueError('History reserve must be between 0 and 1 current region')
        self.prefetch_rings=int(scene.get('regional_prefetch_rings',2))
        if not 1<=self.prefetch_rings<=6:raise ValueError('Use one normal halo and at most five extra prefetch rings')
        super().__init__(scene,*args,**kw)
        self.statistics.update(response_storage='host_library_GPU_regional_residual',
            whole_surface_geometry_recovery=False,noncandidate_geometry_lazy=True,
            omitted_geometry_certified=True,global_operator_device_bytes=0,static_response_equal_load_reuse=False)

    def candidate_columns(self):
        if self.scene.get('regional_device_candidates',False):
            from device_region_candidates import collect
            requested=collect(self)
        else:requested=super().candidate_columns()
        if not self.halo:
            for b in self.bodies:
                fs=np.asarray(b.spec['contact_faces']);adj=[set() for _ in b.X]
                for face in fs:
                    for v in face:adj[v].update(face)
                self.halo.append([np.asarray(sorted(set().union(*(adj[v] for v in f))),dtype=int) for f in fs])
        out=[]
        if not self.retry_required:self.retry_required=[set() for _ in self.bodies]
        for i,(b,ids) in enumerate(zip(self.bodies,requested)):
            nodes=np.unique(ids//3);fs=np.asarray(b.spec['contact_faces'])
            for _ in range(self.prefetch_rings):
                if self.scene.get('regional_device_candidates',False):
                    mask=np.zeros(len(b.X),dtype=bool);mask[nodes]=True;touched=np.any(mask[fs],axis=1)
                else:touched=np.any(np.isin(fs,nodes),axis=1)
                nodes=np.unique(np.r_[nodes,fs[touched].ravel()])
            dofs=(3*nodes[:,None]+np.arange(3)).ravel()
            self.retry_required[i].update(dofs.tolist())
            out.append(np.asarray(sorted(self.retry_required[i]),dtype=np.int64))
        return out

    def refresh_response_window(self):
        if self.incremental_updates:
            from incremental_region_update import update
            result=update(self,requested=getattr(self,'_checkpoint_region_ids',None))
        else:result=super().refresh_response_window()
        if not self.face_ready:
            self.face_ready=[cp.empty(len(b.spec['contact_faces']),dtype=cp.int32) for b in self.bodies]
            self.halo_csr=[(cp.asarray(np.r_[0,np.cumsum([len(h) for h in faces])],dtype=cp.int32),
                            cp.asarray(np.concatenate(faces),dtype=cp.int32)) for faces in self.halo]
        extra_tick=time.perf_counter()
        self.region_epoch+=1
        for i,(b,B,source) in enumerate(zip(self.bodies,self.dynamic,self.sources)):
            if not hasattr(B,'outside_column_bounds'):
                if not hasattr(source,'far_column_bound'):raise ValueError('Precompute the one-ring geometry certificate offline')
                B.outside_column_bounds=cp.asarray(source.far_column_bound)
            if not hasattr(B,'history_age'):B.history_age=np.zeros(len(b.P),dtype=np.int64)
            B.history_age[B.compact_ids]=self.region_epoch
            history=B.window_ids[~np.isin(B.window_ids,B.compact_ids)]
            limit=(len(B.window_slots)-len(B.compact_ids) if self.scene.get('regional_retain_spare',False)
                   else int(self.history_fraction*len(B.compact_ids)))
            if len(history)>limit:
                order=np.argsort(B.history_age[history],kind='stable')
                evict=history[order[:len(history)-limit]]
                B.window_slots[np.isin(B.window_slots,evict)]=-1
                B.window_lookup[cp.asarray(evict)]=-1
                B.window_ids=B.window_slots[B.window_slots>=0]
            self.window_events[-1]['operators'][i]['cached_columns']=len(B.window_ids)
            self.window_events[-1]['operators'][i]['historical_columns']=len(B.window_ids)-len(B.compact_ids)
            ptr,nodes=self.halo_csr[i];n=len(self.face_ready[i])
            self.ready_kernel(((n+127)//128,),(128,),
                (ptr,nodes,B.window_lookup,self.face_ready[i],np.int32(n)))
            # Each loaded node's one-ring is guaranteed by face_ready. Its
            # complementary-row bound was sealed offline and never rescanned here.
        sync();self.window_events[-1]['wall_s']+=time.perf_counter()-extra_tick
        return result

    def restore_response_regions(self,regions):
        """Reload saved exact regions without treating cached history as contact.

        The next GPU candidate/coverage checks still use the restored geometry.
        This initialization-only path does not union unrelated initial-state
        contact with the checkpoint's resident data or expand its halo again.
        """
        if not self.incremental_updates:
            raise ValueError('Exact regional restore requires incremental storage')
        assert len(regions)==len(self.bodies)
        for ids,b in zip(regions,self.bodies):
            assert np.all((ids>=0)&(ids<len(b.P)))
        self._checkpoint_region_ids=[np.unique(ids) for ids in regions]
        self.retry_required=[set() for _ in self.bodies]
        self.window_ready=False
        try:self.rebuild(refresh=True)
        finally:del self._checkpoint_region_ids
        self.retry_required=[set() for _ in self.bodies]

    def rebuild(self,refresh=True):
        if not self.sealed_device:
            self.sealed_device=[RegionalRecovery(self,i) for i in range(len(self.bodies))]
        self.geometry.region_owner=self
        if self.scene.get('regional_compact_rows',False):
            from compact_contact_rows import install
            install(self.geometry)
        if not isinstance(self.resident,SealedContact):self.resident=SealedContact(self)
        if not isinstance(self.graph_working,SeparatedWorkingSet):self.graph_working=SeparatedWorkingSet(self)
        if not hasattr(self,'window_control'):
            source=Path(__file__).with_name('device_globalization.cu').read_text()
            source=source.replace('if(geometry[0]&16)m[5]=16;',
                'if(geometry[0]&16384)m[5]=16384;else if(geometry[0]&8192)m[5]=8192;else if(geometry[0]&16)m[5]=16;')
            self.window_control=cp.RawKernel(source,'control_step');self.control=self.window_control
        # Reuse the exact original region/contact/time equations without uploading the library.
        result=WindowedTimeSurfaceCMS.rebuild(self,refresh)
        return result

    def advance_batch(self,steps):
        if self.scene.get('regional_native_runtime',False):
            if not hasattr(self,'native_runtime'):
                from native_runtime_adapter import NativeRuntime
                self.native_runtime=NativeRuntime(self)
            return self.native_runtime.advance(steps)
        if not 1<=steps<=self.batch_capacity:raise ValueError('Invalid batch size')
        sync();tick=time.perf_counter();accepted=[];completed=0
        try:
            for retry in range(128):
                if self.scene.get('regional_native_batch',False):
                    if not hasattr(self,'native_batch'):
                        from native_region_adapter import NativeBatch
                        self.native_batch=NativeBatch(self)
                    done,error,data=self.native_batch.launch(self,steps-completed)
                else:
                    self.run_state.set(np.asarray([steps-completed,0,0],dtype=np.int32));self.clock.fill(self.t+self.dt)
                    self.trajectory.launch()
                    state=cp.asnumpy(self.run_state);done,error=int(state[1]),int(state[2])
                    data=cp.asnumpy(self.device_history[:done])
                self.batch_launches+=1
                accepted.extend(self.unpack(data));completed+=done
                if done:
                    self.t=float(data[-1,0]);self.retry_required=[set() for _ in self.bodies]
                if not error:break
                self.exceptions+=1
                self.retry_events.append(dict(time_s=self.t,error=error,retry=retry,
                    geometry_flags=int(cp.asnumpy(self.geometry.error)[0]),
                    bound_m=[float(cp.asnumpy(o.last_bound)[0]) for o in self.sealed_device],
                    envelope_m=self.residual_envelope))
                if error in (8192,16,16384):
                    if error==16384:
                        self.bound_retries+=1
                        if self.incremental_updates:
                            measured=max(self.retry_events[-1]['bound_m'])
                            self.residual_envelope=float(np.nextafter(1.25*max(self.residual_envelope,measured),np.inf))
                        else:self.residual_envelope*=2
                        if self.residual_envelope>.01:raise RuntimeError('Residual envelope exceeded 10 mm; no unchecked acceptance')
                        self.geometry.pad=self.residual_envelope
                        self.geometry.skin=self.geometry.base_skin+2*self.residual_envelope
                    if error in (16,16384):
                        # Capacity was exceeded at the unaccepted trial geometry.
                        # Rebuilding at committed x can repeat the same undersized
                        # allocation forever. Only the search anchor is updated;
                        # committed HHT state remains unchanged.
                        self.geometry.refresh(self.mapping.x)
                        self.window_ready=False;self.rebuild(refresh=False)
                    elif self.refresh_response_window():self.rebuild(refresh=False)
                    else:self.graph_working.oldcount.fill(0)
                elif error==105 and self.graph_working.cap<256:
                    old=self.graph_working.cap;self.graph_working.cap*=2;self.graph_working.graph=None
                    self.graph_working.oldkeys=cp.pad(self.graph_working.oldkeys,(0,old));self.graph_working.oldw=cp.pad(self.graph_working.oldw,(0,old))
                    self.rebuild()
                else:raise RuntimeError(f'Regional loop error {error} at {self.t}')
            else:raise RuntimeError('Regional retry budget exceeded')
            assert completed==steps
        finally:self.completed_rows.extend(accepted)
        sync();elapsed=time.perf_counter()-tick
        self.batch_history.append(dict(accepted_steps=completed,end_time_s=self.t,online_s=elapsed))
        self.statistics.update(bound_retries=self.bound_retries,response_window_highwater_bytes=self.window_highwater,
            sealed_device_operator_bytes=0,whole_surface_geometry_recovery=False)
        return accepted,elapsed

    def storage_ledger(self):
        rows=[]
        for b,B in zip(self.bodies,self.dynamic):
            rows.append(dict(surface_dofs=len(B.P),modes=B.P.shape[1],candidate_dofs=len(B.compact_ids),
                capacity=B.C.shape[0],surface_modes_bytes=B.P.nbytes,residual_region_bytes=B.C.nbytes,
                regional_modal_factor_bytes=B.regional_modal_response.nbytes,global_residual_representation_bytes=0))
            rows[-1]['regional_schur_factor_bytes']=getattr(getattr(B,'regional_schur_factor',None),'nbytes',0)
        return dict(bodies=rows,offline_host_library_bytes=sum(s.statistics['stored_bytes'] for s in self.sources),
            global_operator_device_bytes=0,combined_surface_region_matrix_bytes=0,
            full_surface_residual_output_requires_offline_recovery=True,
            residual_envelope_m=self.residual_envelope,history_fraction=self.history_fraction,
            prefetch_rings=self.prefetch_rings,incremental_updates=self.incremental_updates,
            incremental_statistics=getattr(self,'incremental_statistics',None))
