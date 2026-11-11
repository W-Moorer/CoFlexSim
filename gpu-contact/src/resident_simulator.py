"""Shared exact FP64 backend with resident preparation/recovery and temporal seeds."""
import time
import cupy as cp
from device_residual_cms import DeviceResidualCMS
from device_simulator import DeviceMapping
from resident_contact import ResidentContact
from resident_working import ResidentWorkingSet

class ResidentSurfaceCMS(DeviceResidualCMS):
    def __init__(self,scene,bodies,policy=None,runtime_options=None):
        super().__init__(scene,bodies,policy)
        self.runtime_options=dict(reuse_history=True,expansion_width=4);self.runtime_options.update(runtime_options or {})
        self.resident=ResidentContact(self);self.graph_working=ResidentWorkingSet(self,**self.runtime_options)

    def evaluate(self,d,s,t,predictor,velocity_base,reference,timers):
        start=time.perf_counter();h=self.dt
        acc=(d-predictor)/(self.beta*h*h);v=velocity_base+self.gamma*h*acc
        nm,g,scale,free=self.modal_kernels.evaluate(self,d,v,acc,self.old_nonmass,t)
        # Timing is asynchronous here; complete step wall time remains synchronized.
        timers['modal']+=time.perf_counter()-start;start=time.perf_counter()
        mapping=DeviceMapping(self,d,s);x=mapping.x;timers['surface_mapping']+=time.perf_counter()-start;start=time.perf_counter()
        world=self.geometry.build(x);gaps=world.gaps;timers['geometry']+=time.perf_counter()-start;start=time.perf_counter()
        free[self.nd:]=-s
        b=self.resident.prepare(world,mapping,free)
        # Always execute the certified contact path; it also handles separation.
        if len(gaps):
            lam,info=self.graph_working.solve_current(b,world.keys,self.policy['contact_qp_tolerance'])
            direction,targets,targetp,targetd=self.resident.recover(lam)
            self.statistics['contact_restricted_solves']+=info['outer_iterations']
        else:
            lam=cp.zeros(0);direction=free;targets=cp.zeros(self.ns);targetp=targets.copy();targetd=cp.zeros(self.nd)
            self.graph_working.oldcount.fill(0)
        direction[self.nd:]=targets-s
        raw_ref=cp.linalg.norm(g/scale)
        ref=cp.maximum(raw_ref,1e-12) if reference is None else cp.asarray(reference)
        threshold=self.policy['absolute_tolerance']+self.policy['relative_tolerance']*ref
        residual=cp.linalg.norm((g-targetd)/scale);static_error=cp.max(cp.abs(targets-s))
        pen=cp.maximum(0.,-cp.min(gaps)) if len(gaps) else cp.asarray(0.)
        comp=cp.max(cp.abs(lam*gaps)) if len(gaps) else cp.asarray(0.)
        move=cp.max(cp.linalg.norm(mapping.apply(direction).reshape(-1,3),axis=1))
        sigma=cp.min(cp.stack([B.sigma for b,B in zip(self.bodies,self.dynamic) if b.reference])) if any(b.reference for b in self.bodies) else cp.asarray(1.)
        values=cp.asnumpy(cp.stack((residual,static_error,ref,threshold,pen,comp,move,sigma)))
        residual,static_error,reference,threshold,pen,comp,move,sigma=map(float,values)
        if sigma<=0:raise ValueError('Nonpositive reference inertia Schur complement')
        merit=max(residual/threshold,static_error/self.policy['static_tolerance_m'],pen/self.policy['gap_tolerance'],comp/1e-8)
        timers['coupled_solve']+=time.perf_counter()-start
        self.statistics['resident_workspace_rebuilds']=self.resident.rebuilds
        self.statistics['resident_temporal_seed_count']=self.graph_working.temporal_seeds
        self.statistics['resident_working_capacity_growths']=self.graph_working.capacity_growths
        return dict(d=d,s=s,x=x,acc=acc,v=v,nm=nm,world=world,lam=lam,direction=direction,targetp=targetp,
            residual=residual,static_error=static_error,reference=reference,threshold=threshold,pen=pen,comp=comp,merit=merit,move=move)
