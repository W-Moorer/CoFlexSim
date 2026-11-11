"""Physical endpoint contact forces with unweighted algebraic flexibility.

New backend variant; frozen Surface-CMS sources and defaults stay unchanged.
M a + w n_new - alpha n_old = G_d.T lambda; s = R_s G_s.T lambda.
"""
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from surface_cms import (SurfaceCMSSimulator,DynamicBlock,SurfaceMap,SurfaceRows,
                         solve_working_qp)
from consistent_mortar_contact import ContactGeometryError


class PhysicalDynamicBlock(DynamicBlock):
    def __init__(self,b,factor,weight):
        super().__init__(b,factor);self.weight=weight
    def solve(self,rhs):return super().solve(rhs)/self.weight
    def action(self,a):return self.weight*super().action(a)
    def diagonal(self):return self.weight*super().diagonal()


class EndpointSurfaceCMS(SurfaceCMSSimulator):
    def __init__(self,scene,bodies,policy=None):
        super().__init__(scene,bodies,policy)
        self.blocks=[PhysicalDynamicBlock(b,1/(self.beta*self.dt**2*self.w),self.w) for b in bodies]
        self.statistics['contact_force_convention']='physical_endpoint'

    def step(self):
        tick=time.perf_counter();h=self.dt;t=self.t+h;old=self.d.copy();oldv=self.v.copy();olda=self.acc.copy()
        predictor=old+h*oldv+h*h*(.5-self.beta)*olda
        velocity_base=oldv+h*(1-self.gamma)*olda
        d=predictor.copy();s=self.s.copy();p=self.p.copy();reference=None;timers=dict(geometry=0.,coupled_solve=0.,surface_mapping=0.,modal=0.)
        for iteration in range(self.policy['max_iterations']):
            start=time.perf_counter();acc=(d-predictor)/(self.beta*h*h);v=velocity_base+self.gamma*h*acc
            nm=self.nonmass(d,v,t);g=(self.w*nm-self.alpha*self.old_nonmass).copy();sc=[]
            for i,b in enumerate(self.bodies):
                lo,hi=self.doff[i:i+2];guide=b.spec.get('guide_stiffness',0.)+b.spec.get('guide_damping',0.)*self.gamma/(self.beta*h)
                block=self.blocks[i];block.update(d[lo:hi],v[lo:hi],guide)
                g[lo:hi]+=block.mass_action(acc[lo:hi]);sc.extend(block.diagonal())
            scale=np.sqrt(np.maximum(sc,1e-20));timers['modal']+=time.perf_counter()-start
            start=time.perf_counter();mapping=SurfaceMap(self,d,s);x=mapping.x;timers['surface_mapping']+=time.perf_counter()-start
            start=time.perf_counter()
            try:
                gaps,world,_,records=self.geometry.build(x,sp.csr_matrix((x.size,0)))
            except ContactGeometryError:
                if iteration==0 and getattr(self.geometry,'recover_invalid_trials',False):
                    # The extrapolated pose is only an initial iterate. The
                    # last accepted pose is a valid starting point; the HHT
                    # predictor itself stays unchanged in the residual.
                    d=old.copy()
                    timers['geometry']+=time.perf_counter()-start
                    self.statistics['invalid_predictor_restarts']=self.statistics.get('invalid_predictor_restarts',0)+1
                    continue
                raise
            G=SurfaceRows(world,mapping);timers['geometry']+=time.perf_counter()-start
            free=np.r_[np.concatenate([-b.solve(g[self.doff[i]:self.doff[i+1]]) for i,b in enumerate(self.blocks)]),-s]
            start=time.perf_counter();self.system.set_contact_support(world)
            direction,lam,info=solve_working_qp(self,self.system,G,gaps,free,gaps+G@free,
                self.policy['contact_qp_tolerance'],(records,x))
            target=mapping.transpose_action(world.T@lam);targetp=target[self.nd:]
            targets=self.system.full_surface_response(targetp);direction[self.nd:]=targets-s
            residual=float(la.norm((g-target[:self.nd])/scale));static_error=float(np.max(abs(targets-s),initial=0.))
            if reference is None:reference=max(float(la.norm(g/scale)),1e-12)
            threshold=self.policy['absolute_tolerance']+self.policy['relative_tolerance']*reference
            pen=max(0.,-float(gaps.min(initial=0.)));comp=float(np.max(abs(lam*gaps),initial=0.))
            timers['coupled_solve']+=time.perf_counter()-start
            if residual<=threshold and static_error<=self.policy['static_tolerance_m'] and pen<=self.policy['gap_tolerance'] and comp<=1e-8:break
            dd=direction[:self.nd];ds=direction[self.nd:];move=np.max(la.norm((mapping@direction).reshape(-1,3),axis=1),initial=0.)
            step=min(1.,.4*self.geometry.radius/max(move,1e-20))
            for i,b in enumerate(self.bodies):
                if b.revolute:step=min(step,.025/max(abs(dd[self.doff[i]]),1e-20))
            penalty=max(1.,1.5*lam.sum());base_static=.5*p@s;slope=g@dd+p@ds-penalty*pen
            curvature=sum(dd[self.doff[i]:self.doff[i+1]]@block.action(dd[self.doff[i]:self.doff[i+1]]) for i,block in enumerate(self.blocks))
            accepted=False;start=time.perf_counter()
            for backtrack in range(30):
                dt=d+step*dd;st=s+step*ds;pt=p+step*(targetp-p);xt=SurfaceMap(self,dt,st).x
                try:
                    gt=self.geometry.build_gaps(xt)
                except ContactGeometryError:
                    if not getattr(self.geometry,'recover_invalid_trials',False):raise
                    self.statistics['invalid_geometry_trial_rejections']=self.statistics.get('invalid_geometry_trial_rejections',0)+1
                    step*=.5
                    continue
                vp=max(0.,-float(gt.min(initial=0.)))
                change=step*(g@dd)+.5*step*step*curvature+.5*pt@st-base_static+penalty*(vp-pen)
                if change<=1e-4*step*min(slope,0.)+1e-15:
                    d,s,p=dt,st,pt;accepted=True;break
                step*=.5
            timers['geometry']+=time.perf_counter()-start
            if not accepted:raise RuntimeError(f'Endpoint CMS line search failed: {iteration}, {residual=}, {static_error=}, {pen=}')
        else:raise RuntimeError(f'Endpoint CMS iteration limit: {residual=}, {static_error=}, {pen=}')
        self.d=d;self.s=s;self.p=p;self.acc=acc;self.v=v;self.t=t;self.x=x;self.old_nonmass=nm
        self.multiplier_history.commit(records,lam,x)
        nodal=np.asarray(world.T@lam).reshape(-1,3);endpoint=records.endpoint_columns()
        row=dict(t=t,online_s=time.perf_counter()-tick,iterations=iteration+1,residual=residual,threshold=threshold,
            static_equilibrium_error_m=static_error,complementarity_Nm=comp,
            pen_um=float(np.max(-endpoint['minimum'],initial=0.))*1e6,
            force=[nodal[self.x_offsets[i]:self.x_offsets[i+1]].sum(0).tolist() for i in range(len(self.bodies))],
            refs=[float(d[self.doff[i]]) for i,b in enumerate(self.bodies) if b.reference],timers=timers)
        self.last_nodal_force=nodal;self.last_contact=(world,lam,records);self.history.append(row);return row
