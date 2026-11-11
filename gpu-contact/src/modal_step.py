"""Fused retained-coordinate dynamics and surface maps, FP64 throughout."""
from pathlib import Path
import numpy as np
import cupy as cp

class ModalKernels:
    def __init__(self):
        self.module=cp.RawModule(code=Path(__file__).with_suffix('.cu').read_text())
        self.modal=self.module.get_function('modal_step');self.mapping=self.module.get_function('surface_map')
        self.direction=self.module.get_function('surface_direction')

    def evaluate(self,sim,d,v,acc,old,t):
        nm=cp.empty(sim.nd);g=cp.empty(sim.nd);scale=cp.empty(sim.nd);free=cp.empty(sim.nd+sim.ns)
        free[sim.nd:]=-sim.s
        ramp=min(1.,t/max(sim.dt,sim.scene.get('load_ramp',sim.dt)))
        for i,(b,B) in enumerate(zip(sim.bodies,sim.dynamic)):
            lo,hi=sim.doff[i:i+2];n=B.K.shape[0];spec=b.spec
            k=spec.get('guide_stiffness',0.);c=spec.get('guide_damping',0.);B.guide=k+c*sim.gamma/(sim.beta*sim.dt)
            endpoint=getattr(sim,'endpoint_contact',False)
            if endpoint:B.guide*=sim.w
            self.modal((1,),(256,),tuple([B.K,B.M,B.Q,B.D,B.inverse,B.c0,B.d0,B.mt,d[lo:hi],v[lo:hi],acc[lo:hi],old[lo:hi],
                B.c,B.h,B.u,B.I,B.sigma,B.bias,nm[lo:hi],g[lo:hi],scale[lo:hi],free[lo:hi],
                np.int32(n),np.int32(b.offset),np.int32(b.prismatic),np.int32(endpoint)]+[np.float64(z) for z in
                (b.I0,b.total_mass,B.factor,sim.w,sim.alpha,k,c,B.guide,spec.get('guide_target_velocity',0.),
                 spec.get('guide_rest_coordinate',0.),spec.get('torque' if b.revolute else 'reference_force',0.),t,ramp)]),shared_mem=3*n*8)
        return nm,g,scale,free

    def map_body(self,B,q,s,x):
        b=B.source;n=len(B.X);R=cp.empty((3,3));tangent=cp.empty(B.X.size)
        self.mapping(((n+127)//128,),(128,),(B.X,B.P,q,s,B.axis,B.center,R,x,tangent,np.int32(n),np.int32(B.P.shape[1]),
            np.int32(b.offset),np.int32(b.revolute),np.int32(b.prismatic)))
        return R,tangent if b.reference else None

    def apply_body(self,B,dd,ds,R,tangent,out):
        b=B.source;n=len(B.X)
        self.direction(((n+127)//128,),(128,),(B.P,dd,ds,R,tangent if tangent is not None else cp.empty(0),out,
            np.int32(n),np.int32(B.P.shape[1]),np.int32(b.offset)))
