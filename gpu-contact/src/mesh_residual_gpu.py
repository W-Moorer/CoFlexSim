"""Resident exact mesh LDL action. cuSPARSE analysis is initialization only.

No surface inverse, regional square matrix, host library query, or host solve is
used by action(). The retained FE factor is an exact response representation,
not a claim that triangular substitution is faster than a dense lookup.
"""
import numpy as np
import ctypes as ct
import scipy.sparse as sp
import cupy as cp
import cupyx.scipy.sparse as csp
from cupyx.cusparse import SpMatDescriptor, DnMatDescriptor
from cupy_backends.cuda.libs import cusparse as cs

CODE=r'''
extern "C" __global__ void scatter_surface(const double*f,const int*ids,double*x,int n,int nr,int k){
 int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y;
 if(i<n)x[(long long)j*nr+ids[i]]=f[(long long)i*k+j];
}
extern "C" __global__ void gather_surface(const double*x,const int*ids,double*out,int n,int nr,int k){
 int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y;
 if(i<n)out[(long long)i*k+j]=x[(long long)j*nr+ids[i]]-out[(long long)i*k+j];
}
'''

_native=None
_blas=None
_one=np.array(1.,np.float64)
_zero=np.array(0.,np.float64)
GEMM=r'''
extern "C" __global__ void active_product(const double*A,const double*B,double*C,int m,int n,int k,
 long long ar,long long ac,long long br,long long bc,const int*meta){
 if(meta[4]||meta[5])return;int nc=meta[2]-meta[3],lane=threadIdx.x%32;
 for(int q=(blockIdx.x*blockDim.x+threadIdx.x)/32;q<m*nc;q+=gridDim.x*(blockDim.x/32)){
  int i=q/nc,j=meta[3]+q%nc;double v=0.;
  for(int z=lane;z<k;z+=32)v+=A[(long long)i*ar+(long long)z*ac]*B[(long long)z*br+(long long)j*bc];
  for(int d=16;d;d/=2)v+=__shfl_down_sync(0xffffffff,v,d);if(!lane)C[(long long)i*n+j]=v;
 }
}
extern "C" __global__ void matrix_vector(const double*A,const double*B,double*C,int m,int k,long long rs,long long cs){
 int i=blockIdx.x,t=threadIdx.x;double sum=0.;
 for(int q=t;q<k;q+=blockDim.x)sum+=A[(long long)i*rs+(long long)q*cs]*B[q];
 __shared__ double work[128];work[t]=sum;__syncthreads();
 for(int d=64;d;d/=2){if(t<d)work[t]+=work[t+d];__syncthreads();}if(!t)C[i]=work[0];
}
extern "C" __global__ void matrix_product(const double*A,const double*B,double*C,int m,int n,int k,
 long long ar,long long ac,long long br,long long bc,int split){
 int r=blockIdx.x*16+threadIdx.y,c=blockIdx.y*16+threadIdx.x;
 int lo=blockIdx.z*1024,hi=min(k,lo+1024);double sum=0.;
 __shared__ double aa[16][16],bb[16][16];
 for(int q=lo;q<hi;q+=16){
  int a=q+threadIdx.x,b=q+threadIdx.y;
  aa[threadIdx.y][threadIdx.x]=(r<m&&a<hi)?A[(long long)r*ar+(long long)a*ac]:0.;
  bb[threadIdx.y][threadIdx.x]=(c<n&&b<hi)?B[(long long)b*br+(long long)c*bc]:0.;__syncthreads();
  for(int j=0;j<16;j++)sum+=aa[threadIdx.y][j]*bb[j][threadIdx.x];__syncthreads();
 }
 if(r<m&&c<n){if(split>1)atomicAdd(C+(long long)r*n+c,sum);else C[(long long)r*n+c]=sum;}
}
'''
_gemm=None
def matmul(A,B,C,meta=None):
    """FP64 kernels only; valid inside CUDA conditional graph bodies."""
    global _gemm
    if _gemm is None:
        mod=cp.RawModule(code=GEMM);_gemm=tuple(mod.get_function(n) for n in ('matrix_vector','matrix_product','active_product'))
    assert C.flags.c_contiguous
    m,k=A.shape;kk,n=B.shape;assert kk==k and C.shape==(m,n)
    ar,ac=(np.int64(s//8) for s in A.strides)
    if meta is not None:
        br,bc=(np.int64(s//8) for s in B.strides)
        _gemm[2]((min(256,max(1,(m*n+3)//4)),),(128,),
                  (A,B,C,np.int32(m),np.int32(n),np.int32(k),ar,ac,br,bc,meta));return
    if n==1:
        assert B.flags.c_contiguous
        _gemm[0]((m,),(128,),(A,B,C,np.int32(m),np.int32(k),ar,ac));return
    br,bc=(np.int64(s//8) for s in B.strides);split=(k+1023)//1024
    if split>1:C.fill(0)
    _gemm[1](((m+15)//16,(n+15)//16,split),(16,16),
             (A,B,C,np.int32(m),np.int32(n),np.int32(k),ar,ac,br,bc,np.int32(split)))

def native():
    """CUDA supports graph capture; CuPy 13's Python wrapper rejects it."""
    global _native
    if _native is None:
        lib=ct.CDLL('cusparse64_12.dll');p=ct.c_void_p;i=ct.c_int
        lib.cusparseSetStream.argtypes=[p,p];lib.cusparseSetStream.restype=i
        lib.cusparseSpSM_solve.argtypes=[p,i,i,p,p,p,p,i,i,p];lib.cusparseSpSM_solve.restype=i
        _native=lib
    return _native


class CachedSpSM:
    def __init__(self, A, B, X, transpose=False):
        self.handle=cp.cuda.device.get_cusparse_handle()
        self.A=SpMatDescriptor.create(A)
        self.B=DnMatDescriptor.create(B);self.X=DnMatDescriptor.create(X)
        self.A.set_attribute(cs.CUSPARSE_SPMAT_FILL_MODE,cs.CUSPARSE_FILL_MODE_LOWER)
        self.A.set_attribute(cs.CUSPARSE_SPMAT_DIAG_TYPE,cs.CUSPARSE_DIAG_TYPE_UNIT)
        self.desc=cs.spSM_createDescr();self.alpha=np.array(1.,np.float64)
        self.args=(self.handle,cs.CUSPARSE_OPERATION_TRANSPOSE if transpose else cs.CUSPARSE_OPERATION_NON_TRANSPOSE,
                   cs.CUSPARSE_OPERATION_NON_TRANSPOSE,self.alpha.ctypes.data,
                   self.A.desc,self.B.desc,self.X.desc,1,cs.CUSPARSE_SPSM_ALG_DEFAULT,self.desc)
        self.workspace=cp.empty(cs.spSM_bufferSize(*self.args),dtype=cp.int8)
        cs.spSM_analysis(*self.args,self.workspace.data.ptr)

    def solve(self):
        lib=native()
        code=lib.cusparseSetStream(self.handle,cp.cuda.get_current_stream().ptr)
        if code:raise RuntimeError(('cusparseSetStream',code))
        code=lib.cusparseSpSM_solve(*self.args)
        if code:raise RuntimeError(('cusparseSpSM_solve',code))

    def __del__(self):
        if getattr(self,'desc',None):
            try:cs.spSM_destroyDescr(self.desc)
            except Exception:pass


class DeviceMeshResidual:
    def __init__(self, host, surface_modes=None):
        # Scalar CSR is required by this cuSPARSE API. The conversion is exact;
        # storage including indices and library analysis workspaces is measured.
        n=len(host.diagonal)
        LT=sp.bsr_matrix((host.blocks.transpose(0,2,1),host.rows,host.ptr),shape=(n,n))
        L=LT.T.tocsr();L.eliminate_zeros();L.sort_indices()
        self.L=csp.csr_matrix(L)
        self.D=cp.asarray(host.diagonal);self.surface=cp.asarray(host.surface)
        self.P=cp.asarray(host.P) if surface_modes is None else surface_modes
        self.Kinv=cp.asarray(host.Kinv);self.work={}
        self.n=len(host.surface);self.nr=n
        module=cp.RawModule(code=CODE)
        self.scatter=module.get_function('scatter_surface');self.gather=module.get_function('gather_surface')
        self.parameter_bytes=sum(a.nbytes for a in (self.L.data,self.L.indices,self.L.indptr,self.D,self.surface,self.P,self.Kinv))
        self.reserve(1)

    def reserve(self,width):
        if width in self.work:return self.work[width]
        rhs=cp.zeros((self.nr,width),order='F');y=cp.zeros_like(rhs);x=cp.zeros_like(rhs)
        forward=CachedSpSM(self.L,rhs,y);backward=CachedSpSM(self.L,y,x,True)
        f=cp.zeros((self.n,width));out=cp.zeros_like(f);dynamic=cp.zeros_like(f)
        modal=cp.zeros((self.P.shape[1],width));correction=cp.zeros_like(modal)
        w=dict(rhs=rhs,y=y,x=x,f=f,out=out,dynamic=dynamic,modal=modal,correction=correction,forward=forward,backward=backward)
        self.work[width]=w;cp.cuda.get_current_stream().synchronize()
        return w

    def apply(self, f, modal_force=None):
        one=f.ndim==1;width=1 if one else f.shape[1]
        w=self.work[width]  # Initialization must reserve every width before capture.
        F=f.reshape(self.n,width)
        w['rhs'].fill(0)
        self.scatter(((self.n+127)//128,width),(128,),(F,self.surface,w['rhs'],np.int32(self.n),np.int32(self.nr),np.int32(width)))
        w['y'].fill(0);w['x'].fill(0)
        w['forward'].solve();w['y']/=self.D[:,None];w['backward'].solve()
        if modal_force is None:matmul(self.P.T,F,w['modal'])
        else:w['modal'][:]=modal_force.reshape(-1,width)
        matmul(self.Kinv,w['modal'],w['correction'])
        matmul(self.P,w['correction'],w['out'])
        self.gather(((self.n+127)//128,width),(128,),(w['x'],self.surface,w['out'],np.int32(self.n),np.int32(self.nr),np.int32(width)))
        return w['out'][:,0] if one else w['out']

    def storage(self):
        workspace=0;analyses=0
        for w in self.work.values():
            workspace+=sum(v.nbytes for v in w.values() if isinstance(v,cp.ndarray))
            analyses+=w['forward'].workspace.nbytes+w['backward'].workspace.nbytes
        return dict(operator_parameter_bytes=self.parameter_bytes,query_workspace_bytes=workspace,
                    sparse_analysis_bytes=analyses,stored_bytes=self.parameter_bytes+workspace+analyses,
                    dense_surface_or_region_matrix_bytes=0,widths=list(self.work))
