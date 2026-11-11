"""Existing FP64 modal equations with time read from device memory."""
from pathlib import Path
import numpy as np
import cupy as cp
from modal_step import ModalKernels

class ClockModalKernels(ModalKernels):
    def __init__(self):
        source=Path(__file__).with_name('modal_step.cu').read_text()
        marker='double time,double ramp){'
        assert source.count(marker)==1
        source=source.replace(marker,'const double*clock,double ramp_duration){\n double time=clock[0],ramp=fmin(1.,time/ramp_duration);')
        self.module=cp.RawModule(code=source)
        self.modal=self.module.get_function('modal_step');self.mapping=self.module.get_function('surface_map');self.direction=self.module.get_function('surface_direction')

    def evaluate(self,sim,d,v,acc,old,clock):
        nm=cp.empty(sim.nd);g=cp.empty(sim.nd);scale=cp.empty(sim.nd);free=cp.empty(sim.nd+sim.ns)
        for i,(b,B) in enumerate(zip(sim.bodies,sim.dynamic)):
            lo,hi=sim.doff[i:i+2];n=B.K.shape[0];spec=b.spec
            k=spec.get('guide_stiffness',0.);c=spec.get('guide_damping',0.);B.guide=k+c*sim.gamma/(sim.beta*sim.dt)
            endpoint=sim.endpoint_contact
            if endpoint:B.guide*=sim.w
            self.modal((1,),(256,),tuple([B.K,B.M,B.Q,B.D,B.inverse,B.c0,B.d0,B.mt,d[lo:hi],v[lo:hi],acc[lo:hi],old[lo:hi],
                B.c,B.h,B.u,B.I,B.sigma,B.bias,nm[lo:hi],g[lo:hi],scale[lo:hi],free[lo:hi],
                np.int32(n),np.int32(b.offset),np.int32(b.prismatic),np.int32(endpoint)]+[np.float64(z) for z in
                (b.I0,b.total_mass,B.factor,sim.w,sim.alpha,k,c,B.guide,spec.get('guide_target_velocity',0.),
                 spec.get('guide_rest_coordinate',0.),spec.get('torque' if b.revolute else 'reference_force',0.))]+[clock,
                 np.float64(max(sim.dt,sim.scene.get('load_ramp',sim.dt)))]),shared_mem=3*n*8)
        return nm,g,scale,free
