"""Bounded current-region response columns in the shared GPU time loop.

Only C[S,I] is resident, where I covers current candidate load coordinates.
R f = C[S,I] f[I] - P A^-1 P^T f avoids a second response matrix.
All output rows S are retained to update geometry conservatively. The pressure
law and contact solver are unchanged; a missing column causes a retry, never a
zero response. Cache construction, transfer and recapture are online on misses.
"""
import gc,time
from pathlib import Path
import numpy as np
import cupy as cp
from device_simulator import DeviceBlock, sync
from batched_time_simulator import BatchedTimeSurfaceCMS
from aliased_working import AliasedWorkingSet
from fused_region_contact import FusedRegionContact
from device_cached_normal import DeviceCachedNormal
from parallel_modal import ParallelModalKernels
from region_working import CODE as REGION_CODE
from surface_response_source import DenseResidualSource
from response_cache_policy import plan_slots

EXTRA=r'''
extern "C" __global__ void check_window(const int*ptr,const int*col,const double*value,
 const int*lookup,int entries,int rows,int*error){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<entries&&i<ptr[rows]&&value[i]!=0.&&lookup[col[i]]<0)atomicOr(error,8192);
}
extern "C" __global__ void window_response(const double*C,const double*f,const int*ids,const int*count,
 const int*lookup,double*out,int n){
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=n)return;double v=0.;
 for(int j=0;j<count[0];j++){int i=ids[j],c=lookup[i];if(c>=0)v+=C[(long long)c*n+r]*f[i];}
 out[r]=v;
}
'''


def fill_response_columns(B,source,missing,slots,chunk_bytes):
    """Exact FP64 columns; bounded staging instead of a whole-window temporary."""
    n=B.P.shape[0]
    width=max(1,min(64,int(chunk_bytes)//max(8*n,1)))
    peak=0;chunks=0
    for lo in range(0,len(missing),width):
        ids=np.asarray(missing[lo:lo+width]);dst=np.asarray(slots[lo:lo+width])
        host=source.columns(ids)
        device_ids=cp.asarray(ids)
        values=cp.asarray(host)
        values+=B.modal_response@B.P[device_ids].T
        B.C[:,cp.asarray(dst)]=values
        # Commit lookup only after the complete corrected column was uploaded.
        B.window_slots[dst]=ids;B.window_lookup[device_ids]=cp.asarray(dst,dtype=cp.int32)
        peak=max(peak,host.nbytes);chunks+=1
        del host,values,device_ids
    return dict(upload_chunks=chunks,max_staging_array_bytes=peak,chunk_columns=width)


class WindowDeviceBlock(DeviceBlock):
    def initialize_surface_response(self,b):
        self.R=None;self.C=cp.zeros((len(self.P),1),order='F')
        self.window_ids=np.empty(0,dtype=np.int64)
        self.window_lookup=cp.full(len(self.P),-1,dtype=cp.int32)
        self.window_slots=np.full(1,-1,dtype=np.int64)
        # Constant low-rank dynamic term: prepare once on the GPU. Column fills
        # then avoid a large single-core host GEMM on every contact-window miss.
        self.modal_response=self.P@self.inverse


class WindowWorkingSet(AliasedWorkingSet):
    def __init__(self,device,*args,**kw):
        super().__init__(device,*args,**kw)
        code=REGION_CODE.replace('const int*nodes,const int*count){','const int*nodes,const int*count,const int*lookup){')
        old='C[column_major?(long long)col[e]*p+i:(long long)i*p+col[e]]'
        assert code.count(old)==1
        code=code.replace(old,'(lookup[col[e]]>=0?C[(long long)lookup[col[e]]*p+i]:0.)')
        module=cp.RawModule(code=code+EXTRA)
        self.window_kernel=module.get_function('region_response');self.check_kernel=module.get_function('check_window')
        def response(grid,block,args):
            ptr=args[0].data.ptr
            B=next(b for b in device.dynamic if b.C.data.ptr==ptr)
            self.window_kernel(grid,block,tuple(args)+(B.window_lookup,))
        self.region_k['region_response']=response
    def enqueue_prepare(self,b,keys,valid=None):
        super().enqueue_prepare(b,keys,valid)
        for B,(ptr,col,val,_,_,_) in zip(self.device.dynamic,self.buffers):
            self.check_kernel(((len(val)+127)//128,),(128,),(ptr,col,val,B.window_lookup,
                np.int32(len(val)),np.int32(len(b)),self.device.geometry.error))


class WindowContact(FusedRegionContact):
    def __init__(self,sim):
        super().__init__(sim);self.window_response=cp.RawKernel(EXTRA,'window_response')
    def recover(self,lam):
        sim=self.sim;direction=self.free.copy();targets=cp.zeros(sim.ns);forces=cp.zeros(sim.ns);targetd=cp.zeros(sim.nd)
        for i,(b,B,(rows,jr,v),(ids,count)) in enumerate(zip(sim.bodies,sim.dynamic,self.contacts,self.force_storage)):
            sl,sh=sim.soff[i:i+2];dl,dh=sim.doff[i:i+2];n=int(sh-sl)
            f=sim.actions.adjoint(rows,lam);count.fill(0)
            self.pack(((n+127)//128,),(128,),(f,ids,count,np.int32(n)))
            modes=B.P.shape[1];fm=cp.empty(modes)
            if modes:self.sparse_modal((modes,),(128,),(B.P,f,ids,count,fm,np.int32(modes)))
            y0=self.mv(B.inverse,fm);forces[sl:sh]=f;targetd[dl+b.offset:dh]=fm
            y=y0
            if b.reference:
                theta=cp.sum(v*lam)/B.sigma;direction[dl]+=theta;y=y0-B.u*theta;targetd[dl]=cp.sum(jr*lam)
            direction[dl+b.offset:dh]+=y
            self.window_response(((n+127)//128,),(128,),(B.C,f,ids,count,B.window_lookup,targets[sl:sh],np.int32(n)))
            if modes:targets[sl:sh]-=self.mv(B.P,y0)
        return direction,targets,forces,targetd


class WindowedTimeSurfaceCMS(BatchedTimeSurfaceCMS):
    block_type=WindowDeviceBlock
    def __init__(self,scene,bodies,policy=None,batch_capacity=256,sources=None,window_bytes=256*2**20):
        self.sources=sources or [DenseResidualSource(b) for b in bodies]
        self.window_budget=int(window_bytes);self.window_ready=False;self.window_events=[]
        self.window_chunk_bytes=int(scene.get('response_upload_chunk_bytes',32*2**20))
        if not 0<self.window_chunk_bytes<=64*2**20:raise ValueError('Response staging chunk must be in (0, 64 MiB]')
        self.window_highwater=0;self.window_misses=0
        super().__init__(scene,bodies,policy,batch_capacity)

    def candidate_columns(self):
        per=[set() for b in self.bodies]
        for part in self.geometry.parts:
            owners=cp.asnumpy(part.owners);mids=cp.asnumpy(part.mids);valid=(owners>=0)&(mids>=0)
            for body,faces in ((part.slave,owners[valid]),(part.master,mids[valid])):
                if len(faces):per[body].update(np.asarray(self.bodies[body].spec['contact_faces'])[faces].ravel().tolist())
        # A cache exit can occur at a trial state beyond the stored broad-phase
        # region. Include the actual nonzero load support that caused the exit.
        if hasattr(self,'world'):
            used=int(self.world.G.indptr[-1]);col=cp.asnumpy(self.world.G.indices[:used]);val=cp.asnumpy(self.world.G.data[:used])
            col=np.unique(col[val!=0])
            for i in range(len(per)):
                lo,hi=self.soff[i:i+2];per[i].update(((col[(col>=lo)&(col<hi)]-lo)//3).tolist())
        return [(3*np.asarray(sorted(nodes),dtype=np.int64)[:,None]+np.arange(3)).ravel() for nodes in per]

    def refresh_response_window(self):
        sync();start=time.perf_counter();requested=self.candidate_columns()
        fixed=sum(B.modal_response.nbytes for B in self.dynamic)
        required=fixed+sum(len(b.P)*max(1,len(ids))*8 for b,ids in zip(self.bodies,requested))
        if required>self.window_budget:raise MemoryError(f'Current candidate columns require {required} bytes; declared window budget {self.window_budget}')
        # Keep exact material-frame columns in stable slots. Old regions occupy
        # spare slots only; a moving window evicts them before allocating more.
        capacities=[B.C.shape[1] if B.C.shape[1]>=max(1,len(ids)) else
            min(len(b.P),max(1,64*((3*len(ids)//2+63)//64)))
            for b,B,ids in zip(self.bodies,self.dynamic,requested)]
        if fixed+sum(len(b.P)*c*8 for b,c in zip(self.bodies,capacities))>self.window_budget:
            capacities=[max(1,len(ids)) for ids in requested]
        counts=[];changed=False
        for b,B,source,ids,capacity in zip(self.bodies,self.dynamic,self.sources,requested,capacities):
            staging=dict(upload_chunks=0,max_staging_array_bytes=0,chunk_columns=0)
            if B.C.shape[1]!=capacity:
                # Include reusable CuPy pool blocks, but retain a safety reserve
                # for graph/contact buffers and chunked modal corrections.
                free,_=cp.cuda.runtime.memGetInfo();pool=cp.get_default_memory_pool()
                available=free+pool.total_bytes()-pool.used_bytes()
                allocation=len(b.P)*capacity*8
                if allocation+4*self.window_chunk_bytes+512*2**20>available:
                    raise MemoryError(f'Response allocation {allocation} bytes exceeds available GPU headroom {available}; no columns dropped')
                B.C=cp.zeros((len(b.P),capacity),order='F');changed=True
                B.window_slots=np.full(capacity,-1,dtype=np.int64);B.window_lookup.fill(-1)
            missing,free,evicted=plan_slots(B.window_slots,ids)
            if len(missing):
                if len(evicted):B.window_lookup[cp.asarray(evicted)]=-1
                staging=fill_response_columns(B,source,missing,free,self.window_chunk_bytes)
            B.window_ids=B.window_slots[B.window_slots>=0]
            counts.append(dict(surface_dofs=len(b.P),candidate_columns=len(ids),cached_columns=len(B.window_ids),
                new_columns=len(missing),column_capacity=B.C.shape[1],matrix_bytes=B.C.nbytes,
                modal_response_bytes=B.modal_response.nbytes,**staging))
        sync();used=fixed+sum(B.C.nbytes for B in self.dynamic);self.window_highwater=max(self.window_highwater,used)
        self.window_events.append(dict(time_s=self.t,wall_s=time.perf_counter()-start,operators=counts,bytes=used,reallocated=changed))
        self.window_ready=True
        # CSR geometry can stay in place while the response columns change.
        # Invalidate the contact tuples so they cannot retain an old C pointer
        # together with the new column lookup.
        if changed and hasattr(self,'resident'):self.resident.signature=None
        if changed and hasattr(self,'graph_working'):self.graph_working.graph=None
        return changed

    def rebuild(self,refresh=True):
        if not isinstance(self.geometry,DeviceCachedNormal):self.geometry=DeviceCachedNormal(self.bodies,self.scene)
        if not isinstance(self.modal_kernels,ParallelModalKernels):self.modal_kernels=ParallelModalKernels()
        if not isinstance(self.resident,WindowContact):self.resident=WindowContact(self)
        if not isinstance(self.graph_working,WindowWorkingSet):self.graph_working=WindowWorkingSet(self)
        if not hasattr(self,'window_control'):
            source=Path(__file__).with_name('device_globalization.cu').read_text()
            source=source.replace('if(geometry[0]&16)m[5]=16;', 'if(geometry[0]&8192)m[5]=8192;else if(geometry[0]&16)m[5]=16;')
            self.window_control=cp.RawKernel(source,'control_step');self.control=self.window_control
        if refresh or not self.geometry.ready:self.geometry.refresh(self.x)
        if not self.window_ready:self.refresh_response_window()
        return super().rebuild(refresh=False)

    def advance_batch(self,steps):
        if not 1<=steps<=self.batch_capacity:raise ValueError('Invalid batch size')
        sync();start=time.perf_counter();accepted=[];completed=0
        try:
            for retry in range(64):
                self.run_state.set(np.asarray([steps-completed,0,0],dtype=np.int32));self.clock.fill(self.t+self.dt)
                self.trajectory.launch();self.batch_launches+=1
                state=cp.asnumpy(self.run_state);done=int(state[1]);error=int(state[2])
                data=cp.asnumpy(self.device_history[:done]);accepted.extend(self.unpack(data));completed+=done
                if done:self.t=float(data[-1,0])
                if not error:break
                self.exceptions+=1
                if error==8192:
                    self.window_misses+=1
                    if self.refresh_response_window():self.rebuild(refresh=False)
                    else:self.graph_working.oldcount.fill(0)
                elif error==16:
                    self.geometry.refresh(self.x);self.window_ready=False;self.rebuild(refresh=False)
                elif error==105 and self.graph_working.cap<256:
                    old=self.graph_working.cap;self.graph_working.cap*=2;self.graph_working.graph=None
                    self.graph_working.oldkeys=cp.pad(self.graph_working.oldkeys,(0,old));self.graph_working.oldw=cp.pad(self.graph_working.oldw,(0,old));self.rebuild()
                else:raise RuntimeError(f'Window time loop error {error} at {self.t}, accepted {completed}')
            else:raise RuntimeError('Response-window retry budget exceeded')
            assert completed==steps
        finally:self.completed_rows.extend(accepted)
        sync();elapsed=time.perf_counter()-start
        self.batch_history.append(dict(accepted_steps=completed,end_time_s=self.t,online_s=elapsed))
        self.statistics.update(time_batch_launches=self.batch_launches,device_step_exceptions=self.exceptions,
            response_window_misses=self.window_misses,response_window_highwater_bytes=self.window_highwater,
            full_dense_surface_matrices_on_GPU=0,whole_surface_geometry_recovery=True)
        return accepted,elapsed
