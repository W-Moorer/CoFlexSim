"""Unilateral contact LCP solved through global modal and residual compliance."""
import numpy as np
import scipy.linalg as la
from modal_compliance import ModalResidualCompliance

def solve_contact_lcp(C,b,tolerance=1e-10,max_iterations=1000,rank_aware=False,initial=None,prefer_cholesky=False):
    """Bound-constrained convex dual QP. C carries physical compliance units.

    No barrier/penalty spring or positive clearance offset is inserted.
    Singular active subsets are solved by least squares; the returned physical
    gap and complementarity are always checked against the unmodified C.
    """
    n=len(b)
    if not n:return np.zeros(0),{'iterations':0,'minimum_gap':0.,'complementarity':0.}
    C=(C+C.T)*.5;scale=np.sqrt(np.maximum(np.diag(C),1e-30));A=C/scale[:,None]/scale[None,:];f=b/scale
    x=np.zeros(n) if initial is None else np.maximum(np.asarray(initial,float),0.)*scale
    if x.shape!=(n,) or not np.isfinite(x).all():raise ValueError('Invalid dual warm start')
    active=x>0.;null_pivots=0;cholesky_solves=0;least_squares_solves=0
    for iteration in range(max_iterations):
        w=C@(x/scale)+b
        eligible=np.where(~active,w,np.inf)
        if eligible.min()>=-tolerance:
            lam=x/scale
            if np.max(np.abs(w[active]),initial=0.)<=10*tolerance:break
        else:active[np.argmin(eligible)]=True
        for inner in range(2*n+2):
            ids=np.flatnonzero(active);z=np.zeros(n)
            if len(ids):
                block=A[np.ix_(ids,ids)]
                solved=False
                if prefer_cholesky:
                    try:
                        value=la.cho_solve(la.cho_factor(block,check_finite=False),-f[ids],check_finite=False)
                        # Keep the original physical-unit residual test. A
                        # near-singular successful factor can still need SVD.
                        if np.max(abs(scale[ids]*(block@value+f[ids])))<=tolerance:
                            z[ids]=value;solved=True;cholesky_solves+=1
                    except la.LinAlgError:pass
                if not solved:
                    z[ids]=la.lstsq(block,-f[ids],cond=1e-12,check_finite=False)[0];least_squares_solves+=1
                defect=block@z[ids]+f[ids]
                if rank_aware and np.max(np.abs(scale[ids]*defect))>tolerance:
                    eigen,U=la.eigh(block,check_finite=False)
                    N=U[:,eigen<=max(eigen[-1],1.)*1e-12]
                    direction=-(N@(N.T@f[ids]))
                    decreasing=direction<-1e-12*max(np.max(np.abs(direction)),1e-300)
                    if np.any(decreasing):
                        # A singular equality solve with incompatible right
                        # hand side must pivot along its null space until a
                        # multiplier hits zero; repeating least squares stalls.
                        candidates=np.flatnonzero(decreasing)
                        local=candidates[np.argmin(x[ids[candidates]]/-direction[candidates])]
                        step=x[ids[local]]/-direction[local]
                        x[ids]+=step*direction;x[ids]=np.maximum(x[ids],0.)
                        active[ids[local]]=False;x[ids[local]]=0.;null_pivots+=1
                        continue
            negative=active&(z<=0.)
            if not np.any(negative):x=z;break
            ratios=x[negative]/np.maximum(x[negative]-z[negative],1e-300);alpha=min(1.,float(ratios.min()))
            x+=alpha*(z-x);drop=active&(x<=1e-13*max(1.,np.max(x)));active[drop]=False;x[drop]=0.
        else:raise RuntimeError('Contact dual inner active-set stalled')
    else:
        if max_iterations<1 or not rank_aware or not active.any():
            raise RuntimeError('Contact LCP did not converge')
        # A nearly dependent SPD face set can pass the scaled solve but lose
        # micrometre-gap accuracy on undoing the diagonal scaling. Polish
        # the last support in physical units, certifying the original LCP.
        support=np.flatnonzero(active)
        lam=np.zeros(n)
        try:
            block=C[np.ix_(support,support)]
            factor=la.cho_factor(block,check_finite=False)
            lam[support]=la.cho_solve(factor,-b[support],check_finite=False)
        except la.LinAlgError as exc:
            raise RuntimeError('Contact LCP did not converge') from exc
        for _ in range(4):
            w=C@lam+b
            if (lam.min()>=0. and w.min()>=-10*tolerance and
                np.max(abs(w[support]))<=10*tolerance and
                np.min(w[~active],initial=0.)>=-tolerance):
                break
            lam[support]-=la.cho_solve(factor,w[support],check_finite=False)
        else:
            raise RuntimeError('Contact LCP did not converge')
    w=C@lam+b
    if w.min()<-10*tolerance or lam.min()<-1e-12:raise RuntimeError('Contact complementarity check failed')
    return lam,{'iterations':iteration+1,'minimum_gap':float(w.min()),'complementarity':float(np.max(np.abs(lam*w))), 'active_count':int(np.count_nonzero(lam>1e-10)), 'null_space_pivots':null_pivots,'cholesky_solves':cholesky_solves,'least_squares_solves':least_squares_solves}

def modal_contact_direction(model,rhs,G,gaps,interface_inverse):
    """min .5 d.H0.d-rhs.d subject to gaps+G.d>=0, in modal/residual coordinates."""
    G=G.toarray() if hasattr(G,'toarray') else np.asarray(G)
    m=model.V.T@rhs;modal_free=model.V@la.cho_solve(model.factor,m,check_finite=False)
    residual_free=model.inverse(rhs)-modal_free;B=model.V.T@G.T
    free_gap=gaps+G@(modal_free+residual_free)
    if not len(gaps) or free_gap.min()>=0.:
        return modal_free+residual_free,np.zeros(len(gaps)),{'iterations':0,'minimum_gap':float(free_gap.min()) if len(gaps) else None,'complementarity':0.,'active_count':0,'interface_compliance_assembly_skipped_for_feasible_free_step':True}
    modal_columns=model.V@la.cho_solve(model.factor,B,check_finite=False)
    residual_columns=interface_inverse(G.T)-modal_columns
    C_res=G@residual_columns;C_modal=G@modal_columns;C=C_res+C_modal
    lam,info=solve_contact_lcp(C,free_gap)
    a=la.cho_solve(model.factor,m+B@lam,check_finite=False)
    d=model.V@a+residual_free+residual_columns@lam
    info.update(residual_compliance_trace=float(np.trace(C_res)),modal_compliance_trace=float(np.trace(C_modal)))
    return d,lam,info
