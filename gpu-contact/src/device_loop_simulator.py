"""Nested CUDA graph surface CMS: device QP and residual-globalization loops.

Host handles only a completed step or explicit cache/capacity exceptions.
Conservative geometry-cache refresh/recompilation stays in online timing.
"""
import time
import os
from pathlib import Path
import numpy as np
import cupy as cp
from resident_simulator import ResidentSurfaceCMS
from cached_normal_contact import CachedNormalContact
from region_working import RegionWorkingSet
from conditional_graph import NativeGraph
from clock_modal import ClockModalKernels
from device_simulator import DeviceMapping,sync
from direct_region_contact import DirectRegionContact

class DeviceLoopSurfaceCMS(ResidentSurfaceCMS):
    def __init__(self,scene,bodies,policy=None):
        super().__init__(scene,bodies,policy)
        self.geometry=CachedNormalContact(bodies,scene)
        self.resident=DirectRegionContact(self)
        self.modal_kernels=ClockModalKernels();self.graph_working=RegionWorkingSet(self)
        mod=cp.RawModule(code=Path(__file__).with_name('device_globalization.cu').read_text())
        self.control=mod.get_function('control_step');self.commit_kernel=mod.get_function('commit_step');self.skip=mod.get_function('skip_invalid')
        self.record=mod.get_function('record_work');self.timer=mod.get_function('stage_time')
        self.enabled=cp.zeros(1,dtype=cp.int32);self.counts=cp.zeros(4,dtype=cp.int64)
        self.stage_start=cp.zeros(6,dtype=cp.uint64);self.stage_total=cp.zeros(6,dtype=cp.uint64)
        self.profiling=os.environ.get('CONTACT_DEVICE_PROFILE')=='1'
        self.outer=cp.zeros(16,dtype=cp.int32);self.scalar=cp.zeros(3);self.merits=cp.zeros(5);self.report=cp.zeros(7)
        self.clock=cp.zeros(1);self.td=self.d.copy();self.ts=self.s.copy();self.tp=self.p.copy()
        self.ad=cp.empty_like(self.d);self.ass=cp.empty_like(self.s);self.ap=cp.empty_like(self.p);self.adr=cp.empty(self.nd+self.ns)
        self.predictor=cp.empty_like(self.d);self.velocity_base=cp.empty_like(self.v)
        self.rot_ids=cp.asarray([self.doff[i] for i,b in enumerate(bodies) if b.revolute],dtype=cp.int32)
        self.reference_ids=cp.asarray([self.doff[i] for i,b in enumerate(bodies) if b.reference],dtype=cp.int32)
        self.loop=None;self.graph_rebuilds=0;self.exceptions=0
        self.rebuild()

    def mark(self,stage,end):
        if self.profiling:self.timer((1,),(1,),(self.enabled,self.stage_start,self.stage_total,np.int32(stage),np.int32(end)))

    def initialize(self):
        self.outer.fill(0);self.scalar.fill(0);self.merits.fill(0)
        h=self.dt;self.predictor[:]=self.d+h*self.v+h*h*(.5-self.beta)*self.acc
        self.velocity_base[:]=self.v+h*(1-self.gamma)*self.acc
        self.td[:]=self.predictor;self.ts[:]=self.s;self.tp[:]=self.p

    def before_contact(self):
        self.mark(0,0)
        h=self.dt;self.ta=(self.td-self.predictor)/(self.beta*h*h);self.tv=self.velocity_base+self.gamma*h*self.ta
        self.nm,self.g,self.scale,self.free=self.modal_kernels.evaluate(self,self.td,self.tv,self.ta,self.old_nonmass,self.clock)
        self.mapping=DeviceMapping(self,self.td,self.ts)
        self.mark(0,1);self.mark(1,0)
        world=self.geometry.enqueue(self.mapping.x);self.world=world;self.free[self.nd:]=-self.ts
        self.mark(1,1);self.mark(2,0)
        b=self.resident.prepare(world,self.mapping,self.free)
        self.graph_working.enqueue_prepare(b,world.keys,self.geometry.valid)
        self.skip((1,),(1,),(self.geometry.error,self.graph_working.meta))
        self.mark(2,1);self.mark(3,0)

    def after_contact(self):
        self.mark(3,1);self.mark(4,0)
        lam=self.graph_working.enqueue_finish(self.world.keys);self.lam=lam
        self.direction,self.targets,self.targetp,self.targetd=self.resident.recover(lam)
        self.direction[self.nd:]=self.targets-self.ts
        self.mark(4,1);self.mark(5,0)
        residual=cp.linalg.norm((self.g-self.targetd)/self.scale);static_error=cp.max(cp.abs(self.targets-self.ts))
        pen=cp.maximum(0.,-cp.min(self.world.gaps));comp=cp.max(cp.abs(lam*self.world.gaps))
        move=cp.max(cp.linalg.norm(self.mapping.apply(self.direction).reshape(-1,3),axis=1))
        sigma=cp.min(cp.stack([B.sigma for b,B in zip(self.bodies,self.dynamic) if b.reference])) if any(b.reference for b in self.bodies) else cp.asarray(1.)
        metrics=cp.stack((residual,static_error,cp.linalg.norm(self.g/self.scale),pen,comp,move,sigma))
        self.metrics=metrics
        self.control((1,),(256,),(self.outer,self.scalar,self.merits,metrics,self.report,self.geometry.error,self.graph_working.meta,
            self.td,self.ts,self.tp,self.d,self.s,self.ad,self.ass,self.adr,self.ap,self.direction,self.targetp,self.rot_ids,
            np.int32(len(self.rot_ids)),np.int32(self.nd),np.int32(self.ns),np.float64(self.geometry.radius),
            np.float64(self.policy['absolute_tolerance']),np.float64(self.policy['relative_tolerance']),
            np.float64(self.policy['static_tolerance_m']),np.float64(self.policy['gap_tolerance']),np.int32(self.policy['max_iterations'])))
        self.mark(5,1)
        self.record((1,),(1,),(self.enabled,self.graph_working.meta,self.counts,np.int32(self.geometry.statistics['cached_pairs'])))

    def finish(self):
        self.commit_kernel(((max(self.nd,self.ns)+255)//256,),(256,),(self.outer,self.td,self.ts,self.tp,self.tv,self.ta,self.nm,
            self.mapping.x,self.d,self.s,self.p,self.v,self.acc,self.old_nonmass,self.x,np.int32(self.nd),np.int32(self.ns)))
        self.nodal=self.actions.adjoint(self.world.G,self.lam).reshape(-1,3)
        self.force=cp.stack([self.nodal[lo:hi].sum(0) for lo,hi in zip(self.x_offsets[:-1],self.x_offsets[1:])])
        self.output=cp.concatenate((self.outer.astype(cp.float64),self.report,self.world.physical_min.reshape(1),self.force.ravel(),
            self.d[self.reference_ids],self.counts.astype(cp.float64),self.stage_total.astype(cp.float64)))

    def rebuild(self,refresh=True):
        self.enabled.fill(0)
        if refresh:self.geometry.refresh(self.x)
        self.clock.fill(self.t+self.dt);self.initialize()
        # First build the fixed maps/capacities outside capture.
        self.ta=(self.td-self.predictor)/(self.beta*self.dt*self.dt);self.tv=self.velocity_base+self.gamma*self.dt*self.ta
        _,_,_,free=self.modal_kernels.evaluate(self,self.td,self.tv,self.ta,self.old_nonmass,self.clock)
        mapping=DeviceMapping(self,self.td,self.ts);w=self.geometry.enqueue(mapping.x)
        free[self.nd:]=-self.ts;self.resident.prepare(w,mapping,free)
        self.graph_working.ensure(len(w.gaps),self.policy['contact_qp_tolerance'])
        init=NativeGraph.capture(self.initialize,pooled=True)
        pre=NativeGraph.capture(self.before_contact,pooled=True)
        # Initialize a finite trivial QP result for graph construction only.
        self.graph_working.meta.fill(0);self.graph_working.meta[4]=1;self.graph_working.lam.fill(0)
        post=NativeGraph.capture(self.after_contact,pooled=True)
        # Suppress committing warm-up states while capturing the end graph.
        self.outer[5]=99
        end=NativeGraph.capture(self.finish,pooled=True)
        self.loop=NativeGraph.nested([init,pre,self.graph_working.body_graph,post,end],self.outer,self.graph_working.meta)
        self.graph_working.oldcount.fill(0);self.graph_rebuilds+=1;self.enabled.fill(1);sync()

    def step(self):
        sync();tick=time.perf_counter();self.clock.fill(self.t+self.dt)
        for retry in range(8):
            self.loop.launch();out=cp.asnumpy(self.output);state=out[:16].astype(int)
            if not state[5]:break
            self.exceptions+=1
            if state[5]==16:
                if retry:self.geometry.skin*=2
                changed=self.geometry.refresh(self.x)
                if changed or retry:self.rebuild(refresh=False)
            elif state[5]==105 and self.graph_working.cap<256:
                old=self.graph_working.cap;self.graph_working.cap*=2;self.graph_working.graph=None
                self.graph_working.oldkeys=cp.pad(self.graph_working.oldkeys,(0,old));self.graph_working.oldw=cp.pad(self.graph_working.oldw,(0,old));self.rebuild()
            else:raise RuntimeError('Device step failed '+str(state.tolist())+' report '+str(out[16:23].tolist()))
        else:raise RuntimeError('Device step cache/capacity retry limit')
        self.t+=self.dt;vals=out[16:23];self.last_nodal_force=self.nodal
        refstart=24+3*len(self.bodies);refend=refstart+len(self.reference_ids)
        counts=out[refend:refend+4].astype(int);stages=out[refend+4:refend+10]*1e-9
        self.geometry.statistics.update(calls=int(counts[0]),candidate_pairs=int(counts[1]))
        row=dict(t=self.t,iterations=int(state[1]),residual=vals[0],threshold=vals[1],static_equilibrium_error_m=vals[2],
            complementarity_Nm=vals[4],pen_um=max(0.,-out[23])*1e6,force=out[24:24+3*len(self.bodies)].reshape(-1,3).tolist(),
            refs=out[refstart:refend].tolist(),timers={},nonlinear_trial_evaluations=int(state[3]),device_evaluations=int(state[6]))
        if self.profiling:row['device_stage_cumulative_s']=stages.tolist()
        sync();row['online_s']=time.perf_counter()-tick;self.history.append(row)
        self.statistics.update(device_step_graph_rebuilds=self.graph_rebuilds,device_step_exceptions=self.exceptions,
            ordinary_step_host_reads=len(self.history),exception_host_reads=self.exceptions,conditional_outer=True,conditional_contact=True,
            contact_restricted_solves=int(counts[2]),device_response_columns=int(counts[3]))
        reserve=cp.get_default_memory_pool().total_bytes()
        for part in self.loop.refs[:5]:
            reserve+=sum(p.total_bytes() for p in part.refs if isinstance(p,cp.cuda.MemoryPool))
        self.statistics['all_owned_pool_reserved_highwater_bytes']=max(reserve,self.statistics.get('all_owned_pool_reserved_highwater_bytes',0))
        return row
