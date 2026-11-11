"""Positive limited-memory secant inverse around an existing condensed solve.

Used only as an optional acceleration metric. It does not replace the physical
potential, PD global operator, or final contact/stationarity acceptance checks.
"""
import numpy as np


class LimitedMemoryInverse:
    use_regional_contact=False

    def __init__(self,base,history):
        self.base=base;self.di=base.di;self.zi=base.zi;self.history=int(history)
        if self.history<1:raise ValueError('Positive secant history required')
        self.pairs=[];self.previous=None;self.updates=0;self.damped=0;self.skipped=0;self.H=None

    def update(self,H):
        self.base.update(H);self.H=H

    def observe(self,q,gradient):
        if self.previous is not None and not np.array_equal(q,self.previous[0]):
            s=q-self.previous[0];y=gradient-self.previous[1]
            Bs=self.H@s;curvature=float(s@Bs);sy=float(s@y)
            if curvature>1e-30 and np.isfinite(curvature) and np.all(np.isfinite(y)):
                # Powell damping relative to the positive reference metric.
                # Every stored pair has positive curvature, hence each inverse
                # BFGS update preserves positive definiteness.
                if sy<.2*curvature:
                    theta=.8*curvature/(curvature-sy);y=theta*y+(1-theta)*Bs;sy=float(s@y);self.damped+=1
                if sy>1e-30:
                    self.pairs.append((s.copy(),y.copy(),1/sy));self.pairs=self.pairs[-self.history:];self.updates+=1
                else:self.skipped+=1
            else:self.skipped+=1
        self.previous=(q.copy(),gradient.copy())

    def solve(self,rhs):
        vector=rhs.ndim==1;work=np.asarray(rhs).reshape(len(self.di)+len(self.zi),-1).copy();alpha=[]
        for s,y,rho in reversed(self.pairs):
            a=rho*(s@work);alpha.append(a);work-=y[:,None]*a
        out=self.base.solve(work)
        for (s,y,rho),a in zip(self.pairs,reversed(alpha)):
            beta=rho*(y@out);out+=s[:,None]*(a-beta)
        return out[:,0] if vector else out

    def solve_contact(self,rows):
        return self.solve(rows.toarray().T)

    def record(self):
        return dict(updates=self.updates,damped=self.damped,skipped=self.skipped,stored_pairs=len(self.pairs),
                    stored_pair_bytes=sum(s.nbytes+y.nbytes+8 for s,y,_ in self.pairs))
