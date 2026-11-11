"""CUDA surface CMS with the repaired contact and residual globalization."""
import time
import cupy as cp
from device_simulator import DeviceSurfaceCMS,DeviceMapping,sync
from fused_normal_contact import FusedNormalContact
from parallel_working import ParallelWorkingSet
from consistent_mortar_contact import ContactGeometryError


class DeviceResidualCMS(DeviceSurfaceCMS):
    def __init__(self,scene,bodies,policy=None):
        super().__init__(scene,bodies)
        self.policy.update(policy or {})
        self.geometry=FusedNormalContact(bodies,scene)
        self.graph_working=ParallelWorkingSet(self)
        self.geometry.build(self.x);sync()

    def evaluate(self,d,s,t,predictor,velocity_base,reference,timers):
        start=time.perf_counter();h=self.dt
        acc=(d-predictor)/(self.beta*h*h);v=velocity_base+self.gamma*h*acc
        nm,g,scale,free=self.modal_kernels.evaluate(self,d,v,acc,self.old_nonmass,t)
        if any(b.reference and float(B.sigma)<=0. for b,B in zip(self.bodies,self.dynamic)):
            raise ValueError('Nonpositive floating-frame inertia Schur complement')
        sync();timers['modal']+=time.perf_counter()-start;start=time.perf_counter()
        mapping=DeviceMapping(self,d,s);x=mapping.x;sync()
        timers['surface_mapping']+=time.perf_counter()-start;start=time.perf_counter()
        world=self.geometry.build(x);gaps=world.gaps;sync()
        timers['geometry']+=time.perf_counter()-start;start=time.perf_counter()
        free[self.nd:]=-s;b=gaps+self.actions.action(world.G,mapping.apply(free))
        if len(gaps) and float(cp.min(b))<0:
            self.prepare_contact(world,mapping)
            lam,info=self.graph_working.solve(b,self.geometry.seeds(world,b),self.policy['contact_qp_tolerance'])
            direction,targets,targetp,targetd=self.contact_response(lam,free)
            self.statistics['contact_restricted_solves']+=info['outer_iterations']
        else:
            lam=cp.zeros(len(gaps));direction=free;targets=cp.zeros(self.ns)
            targetp=targets.copy();targetd=cp.zeros(self.nd)
        direction[self.nd:]=targets-s
        residual=float(cp.linalg.norm((g-targetd)/scale));static_error=float(cp.max(cp.abs(targets-s)))
        if reference is None:reference=max(float(cp.linalg.norm(g/scale)),1e-12)
        threshold=self.policy['absolute_tolerance']+self.policy['relative_tolerance']*reference
        pen=max(0.,-float(cp.min(gaps))) if len(gaps) else 0
        comp=float(cp.max(cp.abs(lam*gaps))) if len(gaps) else 0
        merit=max(residual/threshold,static_error/self.policy['static_tolerance_m'],pen/self.policy['gap_tolerance'],comp/1e-8)
        move=float(cp.max(cp.linalg.norm(mapping.apply(direction).reshape(-1,3),axis=1)))
        sync();timers['coupled_solve']+=time.perf_counter()-start
        return dict(d=d,s=s,x=x,acc=acc,v=v,nm=nm,world=world,lam=lam,direction=direction,targetp=targetp,
            residual=residual,static_error=static_error,reference=reference,threshold=threshold,
            pen=pen,comp=comp,merit=merit,move=move)

    def step(self):
        sync();tick=time.perf_counter();h=self.dt;t=self.t+h
        predictor=self.d+h*self.v+h*h*(.5-self.beta)*self.acc
        velocity_base=self.v+h*(1-self.gamma)*self.acc
        timers=dict(geometry=0.,coupled_solve=0.,surface_mapping=0.,modal=0.)
        try:e=self.evaluate(predictor.copy(),self.s.copy(),t,predictor,velocity_base,None,timers)
        except ContactGeometryError:e=self.evaluate(self.d.copy(),self.s.copy(),t,predictor,velocity_base,None,timers)
        p=self.p.copy();merits=[];trials=0;trace=[]
        for iteration in range(self.policy['max_iterations']):
            trace.append({k:e[k] for k in ('residual','threshold','static_error','pen','comp','merit')})
            if e['merit']<=1:break
            dd=e['direction'][:self.nd];ds=e['direction'][self.nd:]
            step=min(1.,.4*self.geometry.radius/max(e['move'],1e-20))
            for i,b in enumerate(self.bodies):
                if b.revolute:step=min(step,.025/max(abs(float(dd[self.doff[i]])),1e-20))
            ceiling=max((merits+[e['merit']])[-5:]);accepted=False
            for backtrack in range(30):
                trials+=1
                try:trial=self.evaluate(e['d']+step*dd,e['s']+step*ds,t,predictor,velocity_base,e['reference'],timers)
                except ContactGeometryError:step*=.5;continue
                if trial['merit']<=1 or trial['merit']<=(1-1e-4*step)*ceiling:
                    p=p+step*(e['targetp']-p);merits.append(e['merit']);e=trial;accepted=True;break
                step*=.5
            trace[-1].update(step=step,backtracks=backtrack)
            if not accepted:
                self.last_iteration_trace=trace
                raise RuntimeError('GPU refreshed residual line search failed: '+str(trace[-1]))
        else:
            self.last_iteration_trace=trace
            raise RuntimeError('GPU refreshed residual iteration limit: '+str(trace[-1]))
        self.last_iteration_trace=trace
        self.d=e['d'];self.s=e['s'];self.p=p;self.acc=e['acc'];self.v=e['v'];self.t=t
        self.x=e['x'];self.old_nonmass=e['nm'];self.geometry.commit(e['world'],e['lam'])
        nodal=self.actions.adjoint(e['world'].G,e['lam']).reshape(-1,3)
        solver=self.graph_working
        self.statistics.update(contact_control_host_reads=solver.host_reads,contact_expansions=solver.expansions,
            contact_fallbacks=solver.fallbacks,contact_fallback_reasons=solver.fallback_reasons.copy())
        row=dict(t=t,iterations=iteration+1,residual=e['residual'],threshold=e['threshold'],
            static_equilibrium_error_m=e['static_error'],complementarity_Nm=e['comp'],
            pen_um=max(0.,-float(e['world'].physical_min))*1e6,
            force=cp.asnumpy(cp.stack([nodal[self.x_offsets[i]:self.x_offsets[i+1]].sum(0) for i in range(len(self.bodies))])).tolist(),
            refs=[float(self.d[self.doff[i]]) for i,b in enumerate(self.bodies) if b.reference],timers=timers,
            nonlinear_trial_evaluations=trials)
        sync();row['online_s']=time.perf_counter()-tick
        self.last_nodal_force=nodal;self.history.append(row);return row
