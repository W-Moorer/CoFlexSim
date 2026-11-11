"""Endpoint surface CMS with a refreshed-equilibrium residual line search.

Normal-traction contact is not the total derivative of the moving overlap gap.
Globalize its quasi-Newton step by its actual refreshed coupled residual, rather
than an energy/gap merit function with a different derivative. Final physical
equations, tolerances, modes, residual flexibility and HHT parameters are fixed.
"""
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from endpoint_cms import EndpointSurfaceCMS
from surface_cms import SurfaceMap, SurfaceRows, solve_working_qp
from consistent_mortar_contact import ContactGeometryError
from finite_contact_compliance import solve_contact_subproblem


class ResidualSurfaceCMS(EndpointSurfaceCMS):
    def evaluate(self,d,s,t,predictor,velocity_base,reference,timers):
        start=time.perf_counter();h=self.dt
        acc=(d-predictor)/(self.beta*h*h);v=velocity_base+self.gamma*h*acc
        nm=self.nonmass(d,v,t);g=self.w*nm-self.alpha*self.old_nonmass;sc=[]
        for i,b in enumerate(self.bodies):
            lo,hi=self.doff[i:i+2]
            guide=b.spec.get('guide_stiffness',0.)+b.spec.get('guide_damping',0.)*self.gamma/(self.beta*h)
            block=self.blocks[i];block.update(d[lo:hi],v[lo:hi],guide)
            g[lo:hi]+=block.mass_action(acc[lo:hi]);sc.extend(block.diagonal())
        scale=np.sqrt(np.maximum(sc,1e-20));timers['modal']+=time.perf_counter()-start
        start=time.perf_counter();mapping=SurfaceMap(self,d,s);x=mapping.x
        timers['surface_mapping']+=time.perf_counter()-start;start=time.perf_counter()
        gaps,world,_,records=self.geometry.build(x,sp.csr_matrix((x.size,0)))
        G=SurfaceRows(world,mapping);timers['geometry']+=time.perf_counter()-start
        start=time.perf_counter()
        free=np.r_[np.concatenate([-b.solve(g[self.doff[i]:self.doff[i+1]]) for i,b in enumerate(self.blocks)]),-s]
        self.system.set_contact_support(world)
        direction,lam,info,effective_gaps=solve_contact_subproblem(self,self.system,G,gaps,free,gaps+G@free,
            self.policy['contact_qp_tolerance'],(records,x))
        target=mapping.transpose_action(world.T@lam);targetp=target[self.nd:]
        targets=self.system.full_surface_response(targetp);direction[self.nd:]=targets-s
        residual=float(la.norm((g-target[:self.nd])/scale))
        static_error=float(np.max(abs(targets-s),initial=0.))
        if reference is None:reference=max(float(la.norm(g/scale)),1e-12)
        threshold=self.policy['absolute_tolerance']+self.policy['relative_tolerance']*reference
        pen=max(0.,-float(effective_gaps.min(initial=0.)));comp=float(np.max(abs(lam*effective_gaps),initial=0.))
        merit=max(residual/threshold,static_error/self.policy['static_tolerance_m'],
            pen/self.policy['gap_tolerance'],comp/1e-8)
        move=np.max(la.norm((mapping@direction).reshape(-1,3),axis=1),initial=0.)
        timers['coupled_solve']+=time.perf_counter()-start
        return dict(d=d,s=s,x=x,acc=acc,v=v,nm=nm,world=world,records=records,lam=lam,
            direction=direction,targetp=targetp,residual=residual,static_error=static_error,
            reference=reference,threshold=threshold,pen=pen,comp=comp,merit=merit,move=move)

    def step(self):
        tick=time.perf_counter();h=self.dt;t=self.t+h
        predictor=self.d+h*self.v+h*h*(.5-self.beta)*self.acc
        velocity_base=self.v+h*(1-self.gamma)*self.acc
        timers=dict(geometry=0.,coupled_solve=0.,surface_mapping=0.,modal=0.)
        try:e=self.evaluate(predictor.copy(),self.s.copy(),t,predictor,velocity_base,None,timers)
        except ContactGeometryError:
            e=self.evaluate(self.d.copy(),self.s.copy(),t,predictor,velocity_base,None,timers)
        p=self.p.copy();merits=[];trials=0;trace=[]
        for iteration in range(self.policy['max_iterations']):
            trace.append({k:e[k] for k in ('residual','threshold','static_error','pen','comp','merit')})
            if e['merit']<=1:break
            dd=e['direction'][:self.nd];ds=e['direction'][self.nd:]
            step=min(1.,.4*self.geometry.radius/max(e['move'],1e-20))
            for i,b in enumerate(self.bodies):
                if b.revolute:step=min(step,.025/max(abs(dd[self.doff[i]]),1e-20))
            ceiling=max((merits+[e['merit']])[-getattr(self,'merit_window',5):]);accepted=False
            for backtrack in range(30):
                trials+=1
                try:
                    trial=self.evaluate(e['d']+step*dd,e['s']+step*ds,t,predictor,velocity_base,e['reference'],timers)
                except ContactGeometryError:
                    step*=.5;continue
                if trial['merit']<=1 or trial['merit']<=(1-1e-4*step)*ceiling:
                    p=p+step*(e['targetp']-p);merits.append(e['merit']);e=trial;accepted=True;break
                step*=.5
            trace[-1].update(step=step,backtracks=backtrack)
            if not accepted:
                self.last_iteration_trace=trace
                raise RuntimeError('Refreshed residual line search failed: '+str(trace[-1]))
        else:
            self.last_iteration_trace=trace
            raise RuntimeError('Refreshed residual iteration limit: '+str(trace[-1]))
        self.last_iteration_trace=trace
        self.d=e['d'];self.s=e['s'];self.p=p;self.acc=e['acc'];self.v=e['v'];self.t=t
        self.x=e['x'];self.old_nonmass=e['nm']
        self.multiplier_history.commit(e['records'],e['lam'],e['x'])
        nodal=np.asarray(e['world'].T@e['lam']).reshape(-1,3);endpoint=e['records'].endpoint_columns()
        row=dict(t=t,online_s=time.perf_counter()-tick,iterations=iteration+1,residual=e['residual'],threshold=e['threshold'],
            static_equilibrium_error_m=e['static_error'],complementarity_Nm=e['comp'],
            pen_um=float(np.max(-endpoint['minimum'],initial=0.))*1e6,
            force=[nodal[self.x_offsets[i]:self.x_offsets[i+1]].sum(0).tolist() for i in range(len(self.bodies))],
            refs=[float(self.d[self.doff[i]]) for i,b in enumerate(self.bodies) if b.reference],timers=timers,
            nonlinear_trial_evaluations=trials)
        self.last_nodal_force=nodal;self.last_contact=(e['world'],e['lam'],e['records'])
        self.history.append(row);return row
