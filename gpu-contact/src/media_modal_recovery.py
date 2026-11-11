"""Postprocessing-only volume lifting of cached discrete eigenmode traces.

No eigensolve or online model change. For each cached eigenpair, impose its
known surface values and solve the missing rows of (K - omega^2 M) phi = 0.
The full free-coordinate residual and projected matrices certify the result.
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla


def lift_cached_modes(fe,full_runtime,keep,progress=None,strategy='preconditioned'):
    keep=np.asarray(keep,dtype=int)
    inv=np.full(fe.X.size,-1,dtype=int);inv[fe.free]=np.arange(len(fe.free))
    mapped=inv[full_runtime.original_surface_rows];valid=mapped>=0
    known=mapped[valid];unknown=np.setdiff1d(np.arange(len(fe.free)),known)
    known_values=full_runtime.P[valid][:,keep]
    assert len(np.unique(known))==len(known)
    out=np.zeros((len(fe.free),len(keep)));out[known]=known_values
    K=fe.K.tocsc();mass=fe.M.diagonal();Kuu=K[unknown][:,unknown].tocsc()
    rhs=-(K[unknown][:,known]@known_values)
    Mu=sp.diags(mass[unknown],format='csc');records=[];tick=time.perf_counter()
    pstart=time.perf_counter()
    prelu=sla.splu(Kuu) if strategy=='preconditioned' else None
    pre=sla.LinearOperator(Kuu.shape,matvec=prelu.solve,dtype=np.float64) if prelu is not None else None
    factor_s=time.perf_counter()-pstart
    for j,k in enumerate(keep):
        begin=time.perf_counter();lam=float(full_runtime.K[k,k]/full_runtime.M[k,k])
        A=Kuu-lam*Mu
        iterations=[];direct=strategy=='direct'
        if not direct:
            x,flag=sla.gmres(A,rhs[:,j],x0=prelu.solve(rhs[:,j]),M=pre,rtol=3e-13,atol=0.,
                restart=24,maxiter=1,callback=iterations.append,callback_type='pr_norm')
            direct=flag!=0
        if direct:
            fstart=time.perf_counter()
            lu=sla.splu(A)
            x=lu.solve(rhs[:,j]);x+=lu.solve(rhs[:,j]-A@x)
            if strategy=='preconditioned':
                # Adjacent cached frequencies reuse this exact shifted factor.
                # Every requested equation still uses its own original lambda.
                prelu=lu;pre=sla.LinearOperator(Kuu.shape,matvec=prelu.solve,dtype=np.float64)
                factor_s+=time.perf_counter()-fstart
        out[unknown,j]=x
        ku=K@out[:,j];mu=mass*out[:,j]*lam
        error=float(np.linalg.norm(ku-mu)/max(np.linalg.norm(ku)+np.linalg.norm(mu),1e-30))
        if error>5e-7:raise ValueError(('Cached modal lift failed full FE residual',int(k),error))
        records.append(dict(index=int(k),eigen_residual_relative=error,seconds=time.perf_counter()-begin,
            iterative_steps=len(iterations),direct_factor=direct))
        if progress:progress(j+1,len(keep),records[-1])
    targetK=full_runtime.K[np.ix_(keep,keep)];targetM=full_runtime.M[np.ix_(keep,keep)]
    actualK=out.T@(K@out);actualM=out.T@(mass[:,None]*out)
    ek=float(np.linalg.norm(actualK-targetK)/max(np.linalg.norm(targetK),1e-30))
    em=float(np.linalg.norm(actualM-targetM)/max(np.linalg.norm(targetM),1e-30))
    assert max(ek,em)<1e-6,(ek,em)
    return out,dict(method='cached surface eigenpair trace extension; missing rows of (K-w^2 M) phi=0',
        new_eigenpairs=0,selected_columns=len(keep),known_free_dofs=len(known),unknown_free_dofs=len(unknown),
        full_eigen_residual_max=max(z['eigen_residual_relative'] for z in records),
        projected_stiffness_relative=ek,projected_mass_relative=em,
        wall_s=time.perf_counter()-tick,reused_preconditioner_factor_s=factor_s,strategy=strategy,per_mode=records)
