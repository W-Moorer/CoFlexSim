"""Whole-designated-surface shared response functions with compact near residual.

Rhat = B B^T + (R - B B^T) o (W tensor ones(3,3)), with B an exact
Nystrom factor of the same discrete R, and W the compact C2 Wendland kernel.
Both terms are PSD in exact arithmetic. This extends the sampled multiscale
experiment; it is NOT the previous truncated Q-H-Q^T representation.
No FE solves, local stiffness inversion, force-history fitting or neural network.
"""
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.spatial import cKDTree
from numba import njit


def dofs(ids):return (3*np.asarray(ids)[:,None]+np.arange(3)).ravel()


def farthest_points(x,count):
    first=int(np.argmax(np.linalg.norm(x-x.mean(axis=0),axis=1)));ids=[first]
    d=np.sum((x-x[first])**2,axis=1);d[first]=-1
    while len(ids)<count:
        i=int(np.argmax(d));ids.append(i);d=np.minimum(d,np.sum((x-x[i])**2,axis=1));d[ids]=-1
    return np.asarray(ids)


@njit(cache=False,nogil=True)
def gather_local(ptr,indices,data,rows,cols,out):
    for a in range(len(rows)):
        node=rows[a]//3;component=rows[a]%3
        for b in range(len(cols)):
            cn=cols[b]//3;lo=ptr[node];hi=ptr[node+1]
            while lo<hi:
                mid=(lo+hi)//2
                if indices[mid]<cn:lo=mid+1
                else:hi=mid
            out[a,b]=data[lo,component,cols[b]%3] if lo<ptr[node+1] and indices[lo]==cn else 0.


CODE=r'''
extern "C" __global__ void near_block(const int*ptr,const int*idx,const double*val,
 const long long*rows,const long long*cols,double*out,int nr,int nc){
 long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(p>=(long long)nr*nc)return;
 int r=rows[p/nc],c=cols[p%nc],rn=r/3,cn=c/3,lo=ptr[rn],hi=ptr[rn+1];
 while(lo<hi){int m=(lo+hi)/2;if(idx[m]<cn)lo=m+1;else hi=m;}
 out[p]=(lo<ptr[rn+1]&&idx[lo]==cn)?val[(long long)lo*9+(r%3)*3+c%3]:0.;}
extern "C" __global__ void near_apply(const int*ptr,const int*idx,const double*val,
 const double*f,double*out,int n){
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=n*3)return;int node=r/3,c=r%3;double s=0.;
 for(int j=ptr[node];j<ptr[node+1];++j)for(int k=0;k<3;++k)s+=val[(long long)j*9+c*3+k]*f[3*idx[j]+k];out[r]=s;}
extern "C" __global__ void add_near_panel(const int*ptr,const int*idx,const double*val,
 double*panel,int first,int count,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;int node=first+i;
 for(int j=ptr[node];j<ptr[node+1];++j)for(int a=0;a<3;++a)for(int b=0;b<3;++b)
  panel[(long long)(3*i+a)*(3*n)+3*idx[j]+b]+=val[(long long)j*9+3*a+b];}
extern "C" __global__ void clear_halo(const int*ptr,const int*idx,double*panel,int first,int count,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;int node=first+i;
 for(int j=ptr[node];j<ptr[node+1];++j)for(int a=0;a<3;++a)for(int b=0;b<3;++b)
  panel[(long long)(3*i+a)*(3*n)+3*idx[j]+b]=0.;}
'''


class NearResidual:
    sealed=True
    def __init__(self,n,ptr,indices,data):
        self.n=n;self.ptr=ptr;self.indices=indices;self.data=data;self.packed_query=self
        self.statistics=dict(stored_bytes=ptr.nbytes+indices.nbytes+data.nbytes,block_requests=0,query_s=0.)
        self.matrix=sp.bsr_matrix((data,indices,ptr),shape=(n,n));self.device=None
    def block(self,rows,cols):
        out=np.empty((len(rows),len(cols)));gather_local(self.ptr,self.indices,self.data,np.asarray(rows),np.asarray(cols),out);return out
    def upload(self):
        import cupy as cp
        self.device=(cp.asarray(self.ptr),cp.asarray(self.indices),cp.asarray(self.data));self.k=cp.RawModule(code=CODE)
        self.gather=self.k.get_function('near_block');self.mv=self.k.get_function('near_apply')
    def block_gpu(self,rows,cols):
        import cupy as cp
        if self.device is None:raise RuntimeError('Upload function coefficients before solve')
        rows=cp.asarray(rows,dtype=cp.int64);cols=cp.asarray(cols,dtype=cp.int64);out=cp.empty((len(rows),len(cols)))
        self.gather(((out.size+127)//128,),(128,),(*self.device,rows,cols,out,len(rows),len(cols)));return out


class GlobalLocalCompliance:
    sealed=True
    def __init__(self,x,exact,anchors=128,radius_factor=16.,budget=2300*2**20,progress=None):
        tick=time.perf_counter();self.x=np.asarray(x);self.n=3*len(x);tree=cKDTree(x)
        spacing=tree.query(x,k=2)[0][:,1];self.h=float(np.median(spacing));self.radius=self.h*radius_factor
        self.anchors=farthest_points(x,anchors);a=dofs(self.anchors)
        W=exact(a,a);W=(W+W.T)*.5;L=la.cholesky(W,lower=True)
        # Factor columns are whole-body response functions, not local inverses.
        self.U=np.empty((self.n,len(a)))
        for first in range(0,self.n,768):
            rows=np.arange(first,min(first+768,self.n));v=exact(rows,a)
            self.U[rows]=la.solve_triangular(L,v.T,lower=True).T
        factor_s=time.perf_counter()-tick
        counts=tree.query_ball_point(x,self.radius,return_length=True);pairs=int(counts.sum())
        predicted=self.U.nbytes+pairs*(9*8+4)+(len(x)+1)*4
        if predicted>budget:raise MemoryError(f'Function representation {predicted/2**20:.1f} MiB exceeds per-body RAM/GPU model budget')
        ptr=np.r_[0,np.cumsum(counts)].astype(np.int32);indices=np.empty(pairs,np.int32);values=np.empty((pairs,3,3))
        near_start=time.perf_counter()
        for first in range(0,len(x),32):
            stop=min(first+32,len(x));nodes=np.arange(first,stop)
            neighbors=tree.query_ball_point(x[nodes],self.radius,return_sorted=True);union=np.unique(np.concatenate(neighbors));cols=dofs(union);rows=dofs(nodes)
            residual=exact(rows,cols)-self.U[rows]@self.U[cols].T
            residual=residual.reshape(len(nodes),3,len(union),3).transpose(0,2,1,3)
            for j,node in enumerate(nodes):
                ns=np.asarray(neighbors[j]);where=np.searchsorted(union,ns);s=ptr[node];e=ptr[node+1]
                r=np.linalg.norm(x[ns]-x[node],axis=1)/self.radius;taper=np.maximum(1-r,0)**4*(1+4*r)
                indices[s:e]=ns;values[s:e]=residual[j,where]*taper[:,None,None]
            if progress and (first%512==0 or stop==len(x)):progress(stop,len(x),self.U.nbytes+values.nbytes+indices.nbytes+ptr.nbytes)
        self.source=NearResidual(self.n,ptr,indices,values)
        self.statistics=dict(source='global_Nystrom_functions_plus_PSD_compact_near_residual',
            stored_bytes=predicted,global_factor_bytes=self.U.nbytes,near_bytes=self.source.statistics['stored_bytes'],
            anchors=anchors,rank=len(a),radius_m=self.radius,spacing_m=self.h,radius_factor=radius_factor,
            offline_s=time.perf_counter()-tick,factor_s=factor_s,near_s=time.perf_counter()-near_start,
            node_pairs=pairs,full_surface_coverage=True,block_requests=0,query_s=0.,online_factor_solves=0)
    def block(self,rows,cols):return self.U[rows]@self.U[cols].T+self.source.block(rows,cols)
    def apply(self,f):return self.U@(self.U.T@f)+self.source.matrix@f
    def upload_and_bound(self,faces,progress=None,panel_nodes=128):
        import cupy as cp
        tick=time.perf_counter();self.source.upload();self.device_factor=cp.asarray(self.U)
        n=len(self.x);edges=np.unique(np.array([(int(i),int(j)) for f in faces for i in f for j in f]),axis=0)
        adj=sp.coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(n,n)).tocsr()
        ptr=cp.asarray(adj.indptr,dtype=cp.int32);idx=cp.asarray(adj.indices,dtype=cp.int32)
        add=self.source.k.get_function('add_near_panel');clear=self.source.k.get_function('clear_halo');bound=cp.zeros(self.n)
        for first in range(0,n,panel_nodes):
            count=min(panel_nodes,n-first);v=self.device_factor[3*first:3*(first+count)]@self.device_factor.T
            add(((count+127)//128,),(128,),(*self.source.device,v,first,count,n));clear(((count+127)//128,),(128,),(ptr,idx,v,first,count,n))
            bound=cp.maximum(bound,cp.max(cp.abs(v),axis=0));del v
            if progress and (first%1024==0 or first+count==n):progress(first+count,n)
        cp.cuda.get_current_stream().synchronize();self.far_column_bound=np.nextafter(cp.asnumpy(bound)*(1+1e-12),np.inf)
        self.statistics['upload_and_full_bound_s']=time.perf_counter()-tick
        self.statistics['gpu_parameter_bytes']=self.U.nbytes+self.source.statistics['stored_bytes']
    def apply_gpu(self,f):
        import cupy as cp
        f=cp.asarray(f);out=cp.empty(self.n);self.source.mv(((self.n+127)//128,),(128,),(*self.source.device,f,out,self.n//3))
        return out+self.device_factor@(self.device_factor.T@f)


def algebra_check():
    rng=np.random.default_rng(17);x=rng.normal(size=(22,3));B=rng.normal(size=(66,40));R=B@B.T+np.eye(66)
    def exact(r,c):return R[np.ix_(r,c)]
    model=GlobalLocalCompliance(x,exact,anchors=5,radius_factor=2,budget=2**20)
    M=model.block(np.arange(66),np.arange(66));ev=la.eigvalsh((M+M.T)/2)
    assert ev[0]>0 and la.norm(M-M.T)<1e-10
    assert np.max(abs(np.diag(M)-np.diag(R)))<1e-10
    f=rng.normal(size=66);assert la.norm(model.apply(f)-M@f)<1e-10
    return dict(min_eigenvalue=float(ev[0]),reciprocity=float(la.norm(M-M.T)),diagonal_error=float(abs(np.diag(M)-np.diag(R)).max()))
