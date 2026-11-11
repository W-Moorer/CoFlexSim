"""FP64 contact complementarity on CUDA; host orchestration, no CPU algebra.

The outer set is discovered from every current candidate, not supplied as truth.
Singular equality blocks use a device eigensolve and null-space pivot, without
regularizing the physical contact matrix. Scalar host decisions are measured.
"""
import cupy as cp
from cupyx import errstate
from pathlib import Path

_small_kernel=None


def small_lcp(C,b,tolerance=1e-14,initial=None):
    global _small_kernel
    n=len(b)
    if n>64:return lcp(C,b,tolerance,initial)
    if _small_kernel is None:
        _small_kernel=cp.RawKernel(Path(__file__).with_name('small_qp.cu').read_text(),'small_qp')
    C=cp.ascontiguousarray(C);b=cp.ascontiguousarray(b)
    initial=cp.zeros(n) if initial is None else cp.ascontiguousarray(initial)
    out=cp.empty(n);status=cp.empty(3,dtype=cp.int32)
    import numpy as np
    _small_kernel((1,),(32,),(C,b,initial,out,status,np.int32(n),np.float64(tolerance),np.int32(1000)),
                  shared_mem=(n*n+6*n)*8+2*n*4)
    code,it,count=cp.asnumpy(status)
    if code:
        out,info=lcp(C,b,tolerance,initial);info['device_eigensolver_fallback']=int(code)
        return out,info
    return out,dict(iterations=int(it),active_count=int(count),null_pivots=0,device_eigensolver_fallback=0)


def lcp(C, b, tolerance=1e-14, initial=None, max_iterations=1000):
    n=len(b)
    if not n:
        return cp.zeros(0),dict(iterations=0,active_count=0,null_pivots=0)
    C=(C+C.T)*.5
    scale=cp.sqrt(cp.maximum(cp.diag(C),1e-30))
    A=C/scale[:,None]/scale[None,:];f=b/scale
    x=cp.zeros(n) if initial is None else cp.maximum(initial,0)*scale
    active=x>0;null_pivots=0
    for iteration in range(max_iterations):
        w=C@(x/scale)+b;eligible=cp.where(active,cp.inf,w)
        j=int(cp.argmin(eligible))
        if float(eligible[j])>=-tolerance:
            if float(cp.max(cp.where(active,cp.abs(w),0)))<=10*tolerance:
                break
        else:
            active[j]=True
        for inner in range(2*n+2):
            ids=cp.flatnonzero(active);z=cp.zeros(n)
            if len(ids):
                block=A[ids[:,None],ids[None,:]]
                # cuSOLVER stays on the device. Do not silently accept NaNs or
                # a successful factorization with a large physical-unit defect.
                try:
                    with errstate(linalg='raise'):
                        val=cp.linalg.solve(block,-f[ids])
                    solved=bool(cp.all(cp.isfinite(val))) and float(cp.max(cp.abs(scale[ids]*(block@val+f[ids]))))<=tolerance
                except cp.linalg.LinAlgError:
                    solved=False
                if not solved:
                    ev,U=cp.linalg.eigh(block)
                    keep=ev>cp.maximum(ev[-1],1.)*1e-12
                    coeff=U.T@f[ids]
                    val=-(U@cp.where(keep,coeff/cp.where(keep,ev,1.),0.))
                    defect=block@val+f[ids]
                    if float(cp.max(cp.abs(scale[ids]*defect)))>tolerance:
                        direction=-(U@cp.where(keep,0.,coeff))
                        decreasing=direction<-1e-12*cp.maximum(cp.max(cp.abs(direction)),1e-300)
                        if bool(cp.any(decreasing)):
                            ratios=cp.where(decreasing,x[ids]/cp.maximum(-direction,1e-300),cp.inf)
                            local=int(cp.argmin(ratios));step=ratios[local]
                            x[ids]=cp.maximum(x[ids]+step*direction,0.)
                            active[ids[local]]=False;x[ids[local]]=0.;null_pivots+=1
                            continue
                z[ids]=val
            negative=active&(z<=0.)
            if not bool(cp.any(negative)):
                x=z;break
            ratios=cp.where(negative,x/cp.maximum(x-z,1e-300),cp.inf)
            alpha=cp.minimum(1.,cp.min(ratios));x+=alpha*(z-x)
            drop=active&(x<=1e-13*cp.maximum(1.,cp.max(x)))
            active[drop]=False;x[drop]=0.
        else:
            raise RuntimeError('Device contact inner active set stalled')
    else:
        raise RuntimeError('Device contact LCP iteration limit')
    lam=x/scale;gap=C@lam+b
    if float(cp.min(gap))<-10*tolerance or float(cp.min(lam))<-1e-12:
        raise RuntimeError('Device contact certificate failed')
    return lam,dict(iterations=iteration+1,active_count=int(cp.count_nonzero(lam>1e-10)),null_pivots=null_pivots)


def working_set(columns, free_gap, seeds=None, tolerance=1e-14, capacity=256):
    """columns(ids) returns current W[:,ids]; all arrays and solves are CUDA."""
    m=len(free_gap);lam=cp.zeros(m);b=free_gap
    if not m or float(cp.min(b))>=0:
        return lam,dict(iterations=0,outer_iterations=0,stored_columns=0)
    if isinstance(seeds,tuple):
        ids=cp.asnumpy(seeds[0]).tolist();weights=seeds[1].copy()
    else:
        seeds=seeds or {};ids=list(seeds);weights=cp.asarray([seeds[i] for i in ids])
    j=int(cp.argmin(b))
    if j not in ids:ids.append(j);weights=cp.concatenate((weights,cp.zeros(1)))
    response=None
    total_iterations=0;null_pivots=0
    for outer in range(m):
        if len(ids)>capacity:raise MemoryError('Device contact workspace capacity exceeded')
        added=ids if response is None else ids[-1:]
        block=columns(cp.asarray(added,dtype=cp.int64))
        response=block if response is None else cp.concatenate((response,block),axis=1)
        dids=cp.asarray(ids,dtype=cp.int64)
        weights,info=small_lcp(response[dids],b[dids],tolerance,weights)
        total_iterations+=info['iterations'];null_pivots+=info['null_pivots']
        gap=b+response@weights;worst_active=float(cp.min(gap[dids]))
        eligible=gap.copy();eligible[dids]=cp.inf;j=int(cp.argmin(eligible))
        if float(eligible[j])>=-10*tolerance:
            if worst_active<-100*tolerance:raise RuntimeError('Device outer-set feasibility failed')
            lam[dids]=weights
            return lam,dict(iterations=total_iterations,outer_iterations=outer+1,
                            stored_columns=len(ids),minimum_gap_m=float(cp.min(gap)),null_pivots=null_pivots)
        ids.append(j);weights=cp.concatenate((weights,cp.zeros(1)))
    raise RuntimeError('Device contact working set unresolved')
