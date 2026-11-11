"""Exact fixed-metric surface condensation, separate from frozen baseline.

C0 = R + P (K + c M)^-1 P.T, W = B C0 B.T + vv.T/sigma.
The reference-frame term is evaluated at CURRENT geometry/inertia. This identity
does not discard reference coupling, change HHT, or freeze moving contact maps.
"""
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp


class ContactOperators:
    def __init__(self,snapshot,bodies):
        self.G=sp.csr_matrix(snapshot['G']);self.nd=sum(b.rd for b in bodies)
        self.blocks=[];self.n=self.G.shape[0]
        for i,b in enumerate(bodies):
            dl,dh=snapshot[f'doff{i}'];sl,sh=snapshot[f'soff{i}']
            A=self.G[:,dl:dh].toarray()
            Bfull=self.G[:,self.nd+sl:self.nd+sh]
            ids=np.unique(Bfull.indices);B=Bfull[:,ids]
            P=np.asarray(b.P[ids]);factor=la.cho_factor(snapshot[f'D{i}'],check_finite=False)
            R=b.R[np.ix_(ids,ids)]
            C=R+P@la.cho_solve(factor,P.T,check_finite=False)
            u=snapshot[f'u{i}'];sigma=float(snapshot[f'sigma{i}'])
            if sigma<=0:raise ValueError('Non-positive reference Schur complement')
            v=A[:,0]-A[:,1:]@u
            self.blocks.append(dict(A=A,B=B,C=C,R=R,ids=ids,dl=dl,dh=dh,sl=sl,sh=sh,
                                    factor=factor,u=u,sigma=sigma,v=v))

    def baseline_columns(self,ids=None):
        """Same solve and zero-padded response representation as frozen baseline."""
        out=self.baseline_response(ids)
        W=self.G@out
        return (W+W.T)*.5 if ids is None else W

    def baseline_response(self,ids=None):
        """Return coordinate responses, before the candidate feasibility check."""
        selected=self.G if ids is None else self.G[ids]
        rhs=selected.T.toarray();out=np.zeros_like(rhs)
        for b in self.blocks:
            dl,dh=b['dl'],b['dh'];r=rhs[dl:dh]
            elastic=la.cho_solve(b['factor'],r[1:],check_finite=False)
            # h=D u; equivalently h.T D^-1 r = u.T r.
            theta=(r[0]-b['u']@r[1:])/b['sigma']
            out[dl]=theta;out[dl+1:dh]=elastic-b['u'][:,None]*theta
            idx=self.nd+b['sl']+b['ids']
            out[idx]=b['R']@rhs[idx]
        return out

    def compact_columns(self,ids):
        W=np.zeros((self.n,len(ids)))
        for b in self.blocks:
            W+=b['B']@(b['C']@b['B'][ids].T.toarray())
            W+=b['v'][:,None]*b['v'][ids][None,:]/b['sigma']
        return W

    def compact_cpu(self):
        W=np.zeros((self.n,self.n))
        for b in self.blocks:
            W+=(b['B']@b['C'])@b['B'].T
            W+=np.outer(b['v'],b['v'])/b['sigma']
        return (W+W.T)*.5


class GPUContactOperators:
    def __init__(self,operator,dtype='float64',sparse=True):
        import cupy as cp
        import cupyx.scipy.sparse as csp
        self.cp=cp;self.n=operator.n;self.dtype=cp.dtype(dtype);self.blocks=[]
        for b in operator.blocks:
            B=csp.csr_matrix(b['B'].astype(dtype)) if sparse else cp.asarray(b['B'].toarray(),dtype=dtype)
            self.blocks.append((B,cp.asarray(b['C'],dtype=dtype),
                                cp.asarray(b['v']/np.sqrt(b['sigma']),dtype=dtype)))

    def assemble(self):
        cp=self.cp;W=cp.zeros((self.n,self.n),dtype=self.dtype)
        for B,C,v in self.blocks:
            # Written as sparse-left products, avoiding host or sparse-right fallback.
            BC=B@C
            W+=(B@BC.T).T
            W+=v[:,None]*v[None,:]
        return (W+W.T)*.5

    def columns(self,ids):
        cp=self.cp;W=cp.zeros((self.n,len(ids)),dtype=self.dtype)
        for B,C,v in self.blocks:
            BA=B[ids]
            BA=BA.toarray() if hasattr(BA,'toarray') else BA
            W+=B@(C@BA.T)
            W+=v[:,None]*v[ids][None,:]
        return W


def qp_residual(W,b,lam):
    """Physical constraint residuals; no uniqueness assumption for multipliers."""
    gap=W@lam+b
    return dict(minimum_linearized_gap_m=float(gap.min(initial=0)),
                negative_multiplier_N=float(np.maximum(-lam,0).max(initial=0)),
                complementarity_Nm=float(np.max(np.abs(lam*gap),initial=0)))


class SparseColumnKernel:
    """Two sparse contractions per body; no dense row gathering or BLAS calls.

    T = C B_active.T, then W[:,active] += B T + v v_active.T.
    All candidates are checked through W[:,active], including inactive rows.
    The kernel uses the already constructed contact maps; it does not construct
    or update geometry. Fixed-shape capture can replay with changed values/IDs.
    """
    _compiled={}

    def __init__(self,device,ids,capture=True,warmup=True):
        cp=device.cp;self.cp=cp;self.device=device
        self.ids=cp.asarray(ids,dtype=cp.int64);self.k=len(ids)
        self.out=cp.empty((device.n,self.k),dtype=device.dtype);self.buffers=[]
        scalar='double' if device.dtype==cp.dtype('float64') else 'float'
        column_major=bool(getattr(device,'column_major',False))
        code=r'''
        extern "C" __global__ void response(const SCALAR* C, const int* ptr,
            const int* col,const SCALAR* val,const long long* ids, SCALAR* T,
            const int p,const int k) {
            const int idx=blockIdx.x*blockDim.x+threadIdx.x;
            if(idx>=p*k) return;
            const int i=idx/k,j=idx%k,r=(int)ids[j]; SCALAR sum=0;
            for(int e=ptr[r];e<ptr[r+1];++e) sum+=C[i*p+col[e]]*val[e];
            T[idx]=sum;
        }
        extern "C" __global__ void candidates(const int* ptr,const int* col,
            const SCALAR* val,const SCALAR* v,const long long* ids,
            const SCALAR* T,SCALAR* W,const int n,const int k) {
            const int idx=blockIdx.x*blockDim.x+threadIdx.x;
            if(idx>=n*k) return;
            const int i=idx/k,j=idx%k; SCALAR sum=v[i]*v[ids[j]];
            for(int e=ptr[i];e<ptr[i+1];++e) sum+=val[e]*T[col[e]*k+j];
            W[idx]+=sum;
        }
        '''.replace('SCALAR',scalar)
        if column_major:code=code.replace('C[i*p+col[e]]','C[col[e]*p+i]')
        key=(scalar,column_major)
        if key not in self._compiled:
            self._compiled[key]=(cp.RawKernel(code,'response'),cp.RawKernel(code,'candidates'))
        self.response,self.candidates=self._compiled[key]
        for B,C,v in device.blocks:
            if not hasattr(B,'indptr'):raise ValueError('CSR device maps required')
            if B.indptr.dtype!=cp.int32 or B.indices.dtype!=cp.int32:raise TypeError('32-bit CSR indices required')
            self.buffers.append(cp.empty((C.shape[0],self.k),dtype=device.dtype))
        self.graph=None
        if warmup or capture:self.compute();cp.cuda.get_current_stream().synchronize()
        if capture:
            stream=cp.cuda.Stream(non_blocking=True)
            with stream:
                stream.begin_capture();self.compute();self.graph=stream.end_capture()
            self.graph.launch();cp.cuda.get_current_stream().synchronize()

    def compute(self):
        self.out.fill(0);k=np.int32(self.k);n=np.int32(self.device.n)
        for (B,C,v),T in zip(self.device.blocks,self.buffers):
            p=np.int32(C.shape[0])
            self.response(((int(p)*self.k+255)//256,),(256,),
                (C,B.indptr,B.indices,B.data,self.ids,T,p,k))
            self.candidates(((self.device.n*self.k+255)//256,),(256,),
                (B.indptr,B.indices,B.data,v,self.ids,T,self.out,n,k))
        return self.out

    def replay(self):
        if self.graph is None:return self.compute()
        self.graph.launch();return self.out
