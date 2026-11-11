"""Native trajectory/region runtime, with explicit arena/capture migration exits.

Python prepares models/captured numerical programs. C++ owns accepted-step
advancement, regional requests, history, slot policy, exact refill and envelope
updates. Hybrid mode permits measured Python capacity repair; strict mode does not.
"""
import ctypes as ct
import time
import numpy as np
import cupy as cp
from native_region_adapter import library

u=ct.c_uint64;i=ct.c_int;p=ct.c_void_p;d=ct.c_double;ll=ct.c_int64

class Body(ct.Structure):
    _fields_=[(k,i) for k in ('n','capacity','modes','bank','face_count','node_offset')]+[
        (k,u) for k in ('C','U','P','inverse','D','lookup','ready','halo_ptr','halo_nodes','bound')]+[
        (k,p) for k in ('faces','slots','age','required')]
class Pair(ct.Structure):
    _fields_=[(k,u) for k in ('owners','mids','slave_faces','master_faces')]+[
        (k,i) for k in ('count','slave_offset','master_offset')]
class Config(ct.Structure):
    _fields_=[('bodies',p),('body_count',i),('pairs',p),('pair_count',i)]+[
        (k,u) for k in ('ptr','col','val','oldcount')]+[
        (k,i) for k in ('csr_rows','csr_capacity','node_count','rings')]+[('history_fraction',d),('epoch',ll)]+[
        (k,u) for k in ('envelope','query_radius','force_query')]+[('base_query_radius',d)]
class RegionStats(ct.Structure):
    _fields_=[(k,u) for k in ('updates','new_columns','evicted_columns','device_bytes','host_bytes')]+[('update_s',d)]
class Launch(ct.Structure):
    _fields_=[(k,u) for k in ('exec','stream','run','clock','history')]+[
        ('width',i),('batch_capacity',i),('time',d),('dt',d),('region',p)]
class AdvanceStats(ct.Structure):
    _fields_=[(k,u) for k in ('launches','region_retries','envelope_retries','host_repairs')]+[('solve_s',d),('repair_s',d)]
Repair=ct.CFUNCTYPE(i,p,ct.POINTER(Launch),i,i,p)

class NativeRuntime:
    def __init__(self,sim):
        if sim.scene.get('regional_retain_spare',False) or sim.scene.get('regional_current_pair_filter',False):
            raise ValueError('Native region runtime does not silently substitute unsupported cache policies')
        self.sim=sim;self.lib=library();self.handle=p();self.region_totals={};self.advance_totals={};self.failure=None
        self.repair_events=[];self.references=[];self.callback=Repair(self.repair)
        signatures={
            'cf_region_create':([p,ct.POINTER(Config),ct.POINTER(p)],i),
            'cf_region_destroy':([p],i),'cf_region_update':([p,u,ct.POINTER(i)],i),
            'cf_region_export':([p,i,p,p,p,p,ct.POINTER(ll)],i),
            'cf_region_stats':([p,ct.POINTER(RegionStats)],i),
            'cf_advance':([ct.POINTER(Launch),i,p,ct.POINTER(i),Repair,p,ct.POINTER(AdvanceStats)],i)}
        for name,(args,result) in signatures.items():
            f=getattr(self.lib,name);f.argtypes=args;f.restype=result
        self.bind()
    def check(self,code):
        if code:raise RuntimeError(self.lib.cf_last_error().decode())
    def bind(self):
        sim=self.sim
        if self.handle.value:self.close_region()
        from device_region_candidates import collect
        if not hasattr(sim,'candidate_faces'):collect(sim)
        bodies=[];self.references=[]
        for j,(b,B,source) in enumerate(zip(sim.bodies,sim.dynamic,sim.sources)):
            faces=np.ascontiguousarray(b.spec['contact_faces'],dtype=np.int32)
            req=np.zeros(len(B.P),dtype=np.int32)
            if sim.retry_required[j]:req[list(sim.retry_required[j])]=1
            slots=np.ascontiguousarray(B.window_slots,dtype=np.int64);age=np.ascontiguousarray(B.history_age,dtype=np.int64)
            self.references.extend([faces,req,slots,age])
            base=source.source if hasattr(source,'U') else source
            bank=sim.refill_staging.register(source,base.packed_query)
            arrays=[B.C,B.regional_schur_factor,B.P,B.inverse,B.regional_modal_response,B.window_lookup,
                    sim.face_ready[j],*sim.halo_csr[j],sim.sealed_device[j].last_bound]
            bodies.append(Body(len(B.P),len(B.C),B.P.shape[1],bank,len(faces),int(sim.soff[j]//3),
                *(a.data.ptr for a in arrays),faces.ctypes.data,slots.ctypes.data,age.ctypes.data,req.ctypes.data))
        pairs=[]
        for q in sim.geometry.parts:
            pairs.append(Pair(q.owners.data.ptr,q.mids.data.ptr,sim.candidate_faces[q.slave].data.ptr,
                sim.candidate_faces[q.master].data.ptr,len(q.owners),int(sim.soff[q.slave]//3),int(sim.soff[q.master]//3)))
        self.body_desc=(Body*len(bodies))(*bodies);self.pair_desc=(Pair*len(pairs))(*pairs);G=sim.world.G
        cfg=Config(ct.addressof(self.body_desc),len(bodies),ct.addressof(self.pair_desc),len(pairs),
            G.indptr.data.ptr,G.indices.data.ptr,G.data.data.ptr,sim.graph_working.oldcount.data.ptr,
            G.shape[0],len(G.data),sim.ns//3,sim.prefetch_rings,sim.history_fraction,sim.region_epoch,
            sim.envelope_device.data.ptr,sim.geometry.query_radius_device.data.ptr,sim.geometry.force_query_device.data.ptr,
            sim.geometry.radius+sim.geometry.base_skin)
        self.check(self.lib.cf_region_create(sim.refill_staging.handle,ct.byref(cfg),ct.byref(self.handle)))
    def export(self):
        sim=self.sim
        if not self.handle.value:return
        sim.residual_envelope=float(cp.asnumpy(sim.envelope_device)[0])
        sim.geometry.pad=sim.residual_envelope;sim.geometry.skin=sim.geometry.base_skin+2*sim.residual_envelope
        for j,B in enumerate(sim.dynamic):
            slots=np.empty(len(B.C),dtype=np.int64);age=np.empty(len(B.P),dtype=np.int64)
            required=np.empty(len(B.P),dtype=np.int32);active=np.empty_like(required);epoch=ll()
            self.check(self.lib.cf_region_export(self.handle,j,slots.ctypes.data,age.ctypes.data,
                required.ctypes.data,active.ctypes.data,ct.byref(epoch)))
            B.window_slots=slots;B.window_ids=slots[slots>=0];B.history_age=age;B.compact_ids=np.flatnonzero(active)
            sim.retry_required[j]=set(np.flatnonzero(required).tolist());sim.region_epoch=epoch.value
    def descriptor(self):
        s=self.sim
        return Launch(s.trajectory.exec.value,cp.cuda.get_current_stream().ptr,s.run_state.data.ptr,
            s.clock.data.ptr,s.device_history.data.ptr,s.device_history.shape[1],s.batch_capacity,s.t,s.dt,self.handle)
    def repair(self,user,launch_ptr,error,accepted,rows):
        sim=self.sim;tick=time.perf_counter()
        try:
            self.export();self.close_region();sim.t=launch_ptr.contents.time
            event=dict(time_s=sim.t,error=error,geometry_flags=int(cp.asnumpy(sim.geometry.error)[0]),
                bound_m=[float(cp.asnumpy(o.last_bound)[0]) for o in sim.sealed_device],envelope_m=sim.residual_envelope)
            sim.retry_events.append(event);self.repair_events.append(event);sim.exceptions+=1
            if error in (8192,16,16384):
                if error==16384:
                    sim.bound_retries+=1;measured=max(event['bound_m'])
                    sim.residual_envelope=float(np.nextafter(1.25*max(sim.residual_envelope,measured),np.inf))
                    if sim.residual_envelope>.01:raise RuntimeError('Residual envelope exceeded 10 mm')
                    sim.geometry.pad=sim.residual_envelope;sim.geometry.skin=sim.geometry.base_skin+2*sim.residual_envelope
                    sim.envelope_device.fill(sim.residual_envelope);sim.geometry.query_radius_device.fill(sim.geometry.radius+sim.geometry.skin)
                if error in (16,16384):
                    sim.geometry.refresh(sim.mapping.x);sim.window_ready=False;sim.rebuild(refresh=False)
                elif sim.refresh_response_window():sim.rebuild(refresh=False)
                else:sim.graph_working.oldcount.fill(0)
            elif error==105 and sim.graph_working.cap<256:
                old=sim.graph_working.cap;sim.graph_working.cap*=2;sim.graph_working.graph=None
                sim.graph_working.oldkeys=cp.pad(sim.graph_working.oldkeys,(0,old));sim.graph_working.oldw=cp.pad(sim.graph_working.oldw,(0,old));sim.rebuild()
            else:raise RuntimeError(f'Native runtime error {error} at {sim.t}')
            self.bind();launch_ptr[0]=self.descriptor();event['repair_s']=time.perf_counter()-tick;return 0
        except BaseException as exc:self.failure=exc;return 1
    def close_region(self):
        if self.handle.value:
            stats=RegionStats();self.check(self.lib.cf_region_stats(self.handle,ct.byref(stats)))
            for key,_ in stats._fields_:
                value=getattr(stats,key)
                self.region_totals[key]=(max(self.region_totals.get(key,0),value) if key.endswith('_bytes') else self.region_totals.get(key,0)+value)
            self.check(self.lib.cf_region_destroy(self.handle));self.handle=p()
    def advance(self,steps):
        sim=self.sim;launch=self.descriptor();data=np.empty((steps,launch.width),dtype=np.float64)
        accepted=i();stats=AdvanceStats();tick=time.perf_counter()
        callback=Repair() if sim.scene.get('regional_strict_native',False) else self.callback
        code=self.lib.cf_advance(ct.byref(launch),steps,data.ctypes.data,ct.byref(accepted),callback,None,ct.byref(stats))
        # The ABI's next successful diagnostic call clears its thread-local
        # error. Copy the failing message before exporting accepted metadata.
        native_error=self.lib.cf_last_error().decode() if code else None
        # Even a failed request exposes only accepted states for checkpointing.
        sim.t=launch.time;self.export();rows=sim.unpack(data[:accepted.value]);sim.completed_rows.extend(rows)
        sim.batch_launches+=stats.launches
        for key,_ in stats._fields_:self.advance_totals[key]=self.advance_totals.get(key,0)+getattr(stats,key)
        elapsed=time.perf_counter()-tick
        sim.batch_history.append(dict(accepted_steps=accepted.value,end_time_s=sim.t,online_s=elapsed))
        if self.failure:raise self.failure
        if code:
            self.last_failure=dict(native_error=native_error,code=code,time_s=sim.t,accepted=accepted.value,
                run_state=cp.asnumpy(sim.run_state).tolist(),geometry_error=cp.asnumpy(sim.geometry.error).tolist(),
                queries=[dict(slots=q.slots,maximum_per_face=int(cp.max(q.counts)),
                    total_pairs=int(q.prefix[-1]),pair_capacity=len(part.owners),overflow=int(q.overflow[0]))
                    for q,part in zip(sim.geometry.device_queries,sim.geometry.parts)])
            raise RuntimeError(f'{native_error}; accepted time={sim.t:.17g}; native code={code}')
        if accepted.value!=steps:raise RuntimeError('Incomplete native request')
        return rows,elapsed
    def ledger(self):
        result=dict(self.region_totals)
        if self.handle.value:
            stats=RegionStats();self.check(self.lib.cf_region_stats(self.handle,ct.byref(stats)))
            for key,_ in stats._fields_:result[key]=max(result.get(key,0),getattr(stats,key)) if key.endswith('_bytes') else result.get(key,0)+getattr(stats,key)
        return dict(region=result,advancement=self.advance_totals,repairs=self.repair_events,
            python_repair_boundary=('disabled in strict mode' if self.sim.scene.get('regional_strict_native',False)
                else 'explicit capacity/capture repair only'),standalone_native=False)
    def close(self):self.close_region()
    def __del__(self):
        try:self.close()
        except Exception:pass
