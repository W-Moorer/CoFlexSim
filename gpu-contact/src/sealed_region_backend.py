"""Compact C[I,I] contact solve with offline whole-surface residual recovery.

Same contact, modal equations, reference coupling and convergence tests as the
window backend. No CPU stiffness factor is consulted during online execution.
The sealed residual operator is separately resident, optionally compressed;
its memory must be reported in addition to the candidate square block.
"""
import time
import numpy as np
import cupy as cp
from device_simulator import sync
from region_working import CODE as REGION_CODE
from windowed_surface_backend import WindowDeviceBlock,WindowWorkingSet,WindowContact,WindowedTimeSurfaceCMS
from sealed_surface_gpu import DeviceSealedOperator
from response_cache_policy import plan_slots


class CompactDeviceBlock(WindowDeviceBlock):
    def initialize_surface_response(self,b):
        self.R=None;self.C=cp.zeros((1,1),order='F')
        self.window_ids=np.empty(0,dtype=np.int64)
        self.window_lookup=cp.full(len(self.P),-1,dtype=cp.int32)
        self.window_slots=np.full(1,-1,dtype=np.int64)
        # Dynamic correction is formed on I, not on the entire potential surface.
        self.modal_response=cp.empty((0,0))


class CompactWorkingSet(WindowWorkingSet):
    def surface_size(self,B,C):
        return B.shape[1]

    def __init__(self,device,*args,**kw):
        super().__init__(device,*args,**kw)
        code=REGION_CODE.replace('const int*nodes,const int*count){','const int*nodes,const int*count,const int*lookup){')
        old='C[column_major?(long long)col[e]*p+i:(long long)i*p+col[e]]'
        assert code.count(old)==1
        code=code.replace(old,'((lookup[col[e]]>=0&&lookup[i]>=0)?C[(long long)lookup[col[e]]*p+lookup[i]]:0.)')
        kernel=cp.RawModule(code=code).get_function('region_response')
        def response(grid,block,args):
            B=next(b for b in device.dynamic if b.C.data.ptr==args[0].data.ptr)
            args=list(args);args[7]=np.int32(B.C.shape[0])
            kernel(((B.C.shape[0]+127)//128,),block,tuple(args)+(B.window_lookup,))
        self.region_k['region_response']=response


class SealedContact(WindowContact):
    def recover(self,lam):
        sim=self.sim;direction=self.free.copy();targets=cp.zeros(sim.ns);forces=cp.zeros(sim.ns);targetd=cp.zeros(sim.nd)
        for i,(b,B,(rows,jr,v),(ids,count),op) in enumerate(zip(sim.bodies,sim.dynamic,self.contacts,self.force_storage,sim.sealed_device)):
            sl,sh=sim.soff[i:i+2];dl,dh=sim.doff[i:i+2];n=int(sh-sl)
            f=sim.actions.adjoint(rows,lam);count.fill(0)
            self.pack(((n+127)//128,),(128,),(f,ids,count,np.int32(n)))
            modes=B.P.shape[1];fm=cp.zeros(modes)
            if modes:self.sparse_modal((modes,),(128,),(B.P,f,ids,count,fm,np.int32(modes)))
            y=self.mv(B.inverse,fm);forces[sl:sh]=f;targetd[dl+b.offset:dh]=fm
            if b.reference:
                theta=cp.sum(v*lam)/B.sigma;direction[dl]+=theta;y=y-B.u*theta;targetd[dl]=cp.sum(jr*lam)
            direction[dl+b.offset:dh]+=y
            targets[sl:sh]=op.apply(f,modal_force=fm)
        return direction,targets,forces,targetd


class SealedRegionSurfaceCMS(WindowedTimeSurfaceCMS):
    block_type=CompactDeviceBlock

    def __init__(self,scene,bodies,policy=None,batch_capacity=256,sources=None,window_bytes=256*2**20):
        if sources is None or len(sources)!=len(bodies) or any(not getattr(s,'sealed',False) for s in sources):
            raise ValueError('Precompute and seal all global surface operators before online initialization')
        if any(s.n!=len(b.P) for b,s in zip(bodies,sources)):raise ValueError('Operator/body surface sizes differ')
        self.sealed_device=[];self.sealed_operator_bytes=0
        super().__init__(scene,bodies,policy,batch_capacity,sources=sources,window_bytes=window_bytes)
        self.statistics.update(response_storage='candidate_square_plus_sealed_whole_structure',
            online_stiffness_factor_solves=0,sealed_device_operator_bytes=self.sealed_operator_bytes,
            noncandidate_geometry_lazy=False,static_response_equal_load_reuse=True)

    def region_bytes(self,b,capacity):
        return capacity*capacity*8

    def allocate_region(self,B,capacity):
        B.C=cp.zeros((capacity,capacity),order='F')

    def write_region_columns(self,B,source,all_ids,missing,occupied,destinations):
        a=source.block(all_ids,missing)
        pi=B.P[cp.asarray(all_ids)];pj=B.P[cp.asarray(missing)]
        values=cp.asarray(a)+(pi@B.inverse)@pj.T
        ii,jj=cp.asarray(occupied),cp.asarray(destinations)
        B.C[cp.ix_(ii,jj)]=values;B.C[cp.ix_(jj,ii)]=values.T

    def refresh_response_window(self):
        sync();start=time.perf_counter();requested=self.candidate_columns()
        required=sum(self.region_bytes(b,max(1,len(ids))) for b,ids in zip(self.bodies,requested))
        if required>self.window_budget:raise MemoryError('Candidate C[I,I] exceeds declared region budget')
        capacities=[B.C.shape[0] if B.C.shape[0]>=max(1,len(ids)) else
            min(len(b.P),max(1,64*((3*len(ids)//2+63)//64)))
            for b,B,ids in zip(self.bodies,self.dynamic,requested)]
        if sum(self.region_bytes(b,c) for b,c in zip(self.bodies,capacities))>self.window_budget:
            capacities=[max(1,len(ids)) for ids in requested]
        changed=False;counts=[]
        for b,B,source,ids,capacity in zip(self.bodies,self.dynamic,self.sources,requested,capacities):
            ids=np.unique(ids)
            if B.C.shape!=(capacity,capacity):
                self.allocate_region(B,capacity);B.window_slots=np.full(capacity,-1,dtype=np.int64)
                B.window_lookup.fill(-1);changed=True
            missing,destinations,evicted=plan_slots(B.window_slots,ids)
            if len(missing):
                if len(evicted):B.window_lookup[cp.asarray(evicted)]=-1
                new_slots=B.window_slots.copy();new_slots[destinations]=missing
                occupied=np.flatnonzero(new_slots>=0);all_ids=new_slots[occupied]
                # New columns include every resident region's cross terms.
                self.write_region_columns(B,source,all_ids,missing,occupied,destinations)
                B.window_slots=new_slots
                B.window_lookup[cp.asarray(missing)]=cp.asarray(destinations,dtype=cp.int32)
            B.compact_ids=ids.copy();B.window_ids=B.window_slots[B.window_slots>=0]
            counts.append(dict(surface_dofs=len(b.P),candidate_columns=len(ids),matrix_bytes=B.C.nbytes,
                new_columns=len(missing),cached_columns=len(B.window_ids),capacity=capacity,
                regional_modal_factor_bytes=getattr(getattr(B,'regional_modal_response',None),'nbytes',0),
                online_factor_solves=0,sealed_operator_bytes=source.statistics['stored_bytes']))
        sync();used=sum(self.region_bytes(b,B.C.shape[0]) for b,B in zip(self.bodies,self.dynamic))
        self.window_highwater=max(self.window_highwater,used)
        self.window_events.append(dict(time_s=self.t,wall_s=time.perf_counter()-start,operators=counts,
            bytes=used,sealed_device_operator_bytes=self.sealed_operator_bytes,reallocated=changed))
        self.window_ready=True
        if changed and hasattr(self,'resident'):self.resident.signature=None
        if changed and hasattr(self,'graph_working'):self.graph_working.graph=None
        return changed

    def rebuild(self,refresh=True):
        if not self.sealed_device:
            self.sealed_device=[DeviceSealedOperator(s,surface_modes=B.P) for s,B in zip(self.sources,self.dynamic)]
            self.sealed_operator_bytes=sum(s.stored_bytes for s in self.sealed_device)
        if not isinstance(self.resident,SealedContact):self.resident=SealedContact(self)
        if not isinstance(self.graph_working,CompactWorkingSet):self.graph_working=CompactWorkingSet(self)
        return super().rebuild(refresh)
