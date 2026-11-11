"""Batched direct nested-compliance application, never a stiffness solve.

All parameters are resident. Inputs/target lists and new working-set columns
are selected on the GPU. Full-surface recovery is supported by the same path.
This module does not build or store an R_II matrix or refill from the CPU.
"""
import numpy as np
import cupy as cp
from nested_resident_gpu import NestedResidentGPU

CODE=r'''
__device__ bool active(const int*m,int j){return !m[4]&&!m[5]&&j>=m[3]&&j<m[2];}
extern "C" __global__ void clear(double*out,double*q,double*y,double*mq,double*mp,
 int*fc,int*count,int n,int nq,int nc,int mr,int w,const int*m){
 int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y;if(!active(m,j))return;
 if(i<n)out[(long long)i*w+j]=0.;if(i<nq){q[(long long)i*w+j]=0.;y[(long long)i*w+j]=0.;}
 if(i<mr){mq[(long long)i*w+j]=0.;mp[(long long)i*w+j]=0.;}
 if(i<nc)fc[(long long)i*w+j]=0;if(i==0)count[j]=0;
}
extern "C" __global__ void source(const int*ip,const int*ic,const int*ir,const int*ptr,
 const double*f,int*fc,int*fi,int n,int w,const int*m){
 int g=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y;if(g>=n||!active(m,j)||f[(long long)g*w+j]==0.)return;
 for(int k=ip[g];k<ip[g+1];k++){int c=ic[k],z=atomicAdd(fc+(long long)c*w+j,1);
 fi[((long long)ptr[c]+z)*w+j]=ir[k];}
}
extern "C" __global__ void targets(const int*ip,const int*ic,const int*ir,const int*ptr,
 const int*list,const int*nt,int*tc,int*ti,int n,int full){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=(full?n:*nt))return;int g=full?i:list[i];
 for(int k=ip[g];k<ip[g+1];k++){int c=ic[k];ti[ptr[c]+atomicAdd(tc+c,1)]=ir[k];}
}
extern "C" __global__ void up(const int*leaves,int nl,const int*parent,const int*rank,
 const long long*bp,const long long*ep,const double*B,const double*T,const int*ids,const int*ptr,
 const int*fc,const int*fi,const double*f,double*q,int maxr,int width,const int*m){
 __shared__ double v[128],s[128];int l=blockIdx.x,r=threadIdx.x,j=blockIdx.y;
 if(l>=nl||!active(m,j))return;int c=leaves[l];if(!fc[(long long)c*width+j]||!rank[c])return;
 double a=0.;if(r<rank[c]){for(int k=0;k<fc[(long long)c*width+j];k++){
 int row=fi[((long long)ptr[c]+k)*width+j];a+=B[bp[c]+(long long)row*rank[c]+r]*f[(long long)ids[ptr[c]+row]*width+j];}
 v[r]=a;atomicAdd(q+((long long)c*maxr+r)*width+j,a);}__syncthreads();
 while(parent[c]>=0){int p=parent[c];if(!rank[p])break;a=0.;
 if(r<rank[p]){for(int k=0;k<rank[c];k++)a+=T[ep[c]+(long long)k*rank[p]+r]*v[k];
 s[r]=a;atomicAdd(q+((long long)p*maxr+r)*width+j,a);}__syncthreads();
 if(r<rank[p])v[r]=s[r];__syncthreads();c=p;}
}
extern "C" __global__ void schedule(const int*aa,const int*bb,int nb,const int*fc,
 const int*tc,int*count,int*tasks,int width,const int*m){
 int z=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y;if(z>=nb||!active(m,j))return;
 int a=aa[z],b=bb[z];if(tc[a]&&fc[(long long)b*width+j])tasks[(long long)atomicAdd(count+j,1)*width+j]=2*z;
 if(a!=b&&tc[b]&&fc[(long long)a*width+j])tasks[(long long)atomicAdd(count+j,1)*width+j]=2*z+1;
}
extern "C" __global__ void couple(const int*aa,const int*bb,const int*kind,const int*lr,const long long*vp,
 const int*ptr,const int*rank,const int*fc,const int*tc,const int*fi,const int*ti,const int*ids,
 const double*V,const double*f,const double*q,double*y,double*out,int maxr,
 const int*count,const int*tasks,int width,const int*m){
 __shared__ double tmp[128];int col=blockIdx.y;if(!active(m,col))return;
 for(int task=blockIdx.x;task<count[col];task+=gridDim.x){
 int id=tasks[(long long)task*width+col],z=id/2,rev=id%2,a=aa[z],b=bb[z],dest=rev?b:a,src=rev?a:b;
 int na=ptr[a+1]-ptr[a],nb=ptr[b+1]-ptr[b],ns=fc[(long long)src*width+col];
 if(kind[z]==1){int r=threadIdx.x;if(r<rank[dest]){double v=0.;for(int j=0;j<rank[src];j++)
 v+=V[vp[z]+(rev?(long long)j*rank[b]+r:(long long)r*rank[b]+j)]*q[((long long)src*maxr+j)*width+col];
 atomicAdd(y+((long long)dest*maxr+r)*width+col,v);}}
 else if(kind[z]==2){int rk=lr[z],r=threadIdx.x;long long sb=vp[z]+(rev?0:(long long)na*rk);
 if(r<rk){double v=0.;for(int k=0;k<ns;k++){int j=fi[((long long)ptr[src]+k)*width+col];
 v+=V[sb+(long long)j*rk+r]*f[(long long)ids[ptr[src]+j]*width+col];}tmp[r]=v;}
 __syncthreads();long long db=vp[z]+(rev?(long long)na*rk:0);
 for(int k=threadIdx.x;k<tc[dest];k+=blockDim.x){int j=ti[ptr[dest]+k];double v=0.;
 for(int r=0;r<rk;r++)v+=V[db+(long long)j*rk+r]*tmp[r];atomicAdd(out+(long long)ids[ptr[dest]+j]*width+col,v);}}
 else{for(int k=threadIdx.x;k<tc[dest];k+=blockDim.x){int r=ti[ptr[dest]+k];double v=0.;
 for(int l=0;l<ns;l++){int j=fi[((long long)ptr[src]+l)*width+col];
 v+=V[vp[z]+(rev?(long long)j*nb+r:(long long)r*nb+j)]*f[(long long)ids[ptr[src]+j]*width+col];}
 atomicAdd(out+(long long)ids[ptr[dest]+r]*width+col,v);}}
 __syncthreads();}
}
extern "C" __global__ void down(const int*leaves,int nl,const int*parent,const int*rank,
 const long long*bp,const long long*ep,const double*B,const double*T,const int*ids,const int*ptr,
 const int*tc,const int*ti,const double*y,double*out,int maxr,int width,const int*m){
 __shared__ double v[128],s[128];int l=blockIdx.x,r=threadIdx.x,col=blockIdx.y;
 if(l>=nl||!active(m,col))return;int leaf=leaves[l];if(!tc[leaf]||!rank[leaf])return;
 int path[32],depth=0,c=leaf;while(c>=0&&depth<32){path[depth++]=c;c=parent[c];}
 for(int j=depth-1;j>=0;j--){c=path[j];int p=parent[c];double a=0.;
 if(r<rank[c]){a=y[((long long)c*maxr+r)*width+col];if(p>=0&&rank[p])
 for(int k=0;k<rank[p];k++)a+=T[ep[c]+(long long)r*rank[p]+k]*v[k];s[r]=a;}
 __syncthreads();if(r<rank[c])v[r]=s[r];__syncthreads();}
 for(int k=r;k<tc[leaf];k+=blockDim.x){int j=ti[ptr[leaf]+k];double a=0.;
 for(int z=0;z<rank[leaf];z++)a+=B[bp[leaf]+(long long)j*rank[leaf]+z]*v[z];
 out[(long long)ids[ptr[leaf]+j]*width+col]+=a;}
}
extern "C" __global__ void modal_up(const double*P,int nr,const int*fc,const int*fi,const int*ids,
 const double*f,double*q,int width,const int*m){
 int col=blockIdx.y;if(!active(m,col))return;int r=blockIdx.x,l=threadIdx.x;
 double v=0.;for(int k=l;k<fc[col];k+=128){int g=ids[fi[(long long)k*width+col]];
 v+=P[(long long)g*nr+r]*f[(long long)g*width+col];}
 __shared__ double a[128];a[l]=v;__syncthreads();for(int d=64;d;d/=2){if(l<d)a[l]+=a[l+d];__syncthreads();}
 if(!l)q[(long long)r*width+col]=a[0];
}
extern "C" __global__ void modal_matrix(const double*K,const double*q,double*y,int n,int width,const int*m){
 int j=blockIdx.y;if(!active(m,j))return;int r=blockIdx.x,l=threadIdx.x;double a=0.;
 for(int k=l;k<n;k+=32)a+=K[(long long)r*n+k]*q[(long long)k*width+j];
 for(int s=16;s;s/=2)a+=__shfl_down_sync(0xffffffff,a,s);if(!l)y[(long long)r*width+j]=a;
}
extern "C" __global__ void finish(const int*targets,const int*nt,int full,const double*f,const double*d,
 const double*P,const double*mq,int nr,double*out,int n,int width,const int*m){
 int j=blockIdx.y;if(!active(m,j))return;int k=blockIdx.x*4+threadIdx.x/32,l=threadIdx.x%32;
 if(k>=(full?n:*nt))return;int g=full?k:targets[k];double a=0.;
 for(int z=l;z<nr;z+=32)a+=P[(long long)g*nr+z]*mq[(long long)z*width+j];
 for(int s=16;s;s/=2)a+=__shfl_down_sync(0xffffffff,a,s);
 if(!l)out[(long long)g*width+j]+=d[g]*f[(long long)g*width+j]-a;
}
'''


class DirectResponseGPU:
    def __init__(self,model,P,Kinv,surface_modes=None):
        # Reuse the already-validated immutable nested parameter layout only.
        tmp=NestedResidentGPU(model,P,Kinv,[(np.array([0]),np.zeros(1),np.array([0]))])
        self.params=tmp.params
        if surface_modes is not None:self.params['P']=surface_modes
        self.n=tmp.n;self.nc=tmp.nc;self.nr=tmp.nr;self.maxr=tmp.maxr
        self.k={name:cp.RawKernel(CODE,name) for name in ['clear','source','targets','up','schedule','couple','down','modal_up','modal_matrix','finish']}
        for k,v in self.params.items():setattr(self,k,v)
        self.parameter_bytes=sum(v.nbytes for v in self.params.values())
        self.work={};self.reserve(1)

    def reserve(self,width):
        width=int(width)
        if width not in self.work:
            if width>1:
                cp.cuda.get_current_stream().synchronize()
                for old in list(self.work):
                    if old>1:del self.work[old]
            self.work[width]=dict(f=cp.zeros((self.n,width)),out=cp.zeros((self.n,width)),
                q=cp.zeros((self.nc*self.maxr,width)),y=cp.zeros((self.nc*self.maxr,width)),
                mq=cp.zeros((self.nr,width)),mp=cp.zeros((self.nr,width)),
                fc=cp.zeros((self.nc,width),dtype=cp.int32),fi=cp.empty((len(self.ids),width),dtype=cp.int32),
                tc=cp.zeros(self.nc,dtype=cp.int32),ti=cp.empty(len(self.ids),dtype=cp.int32),
                count=cp.zeros(width,dtype=cp.int32),tasks=cp.empty((2*len(self.aa),width),dtype=cp.int32),
                default_meta=cp.asarray([0,0,width,0,0,0,0],dtype=cp.int32),
                no_targets=cp.zeros(1,dtype=cp.int32),no_count=cp.zeros(1,dtype=cp.int32))
        return self.work[width]

    def apply(self,f,modal_force=None,meta=None,modal_inverse=None,targets=None):
        one=f.ndim==1;width=1 if one else f.shape[1];w=self.work[width]
        if meta is None:meta=w['default_meta']
        F=f.reshape(self.n,width);nq=self.nc*self.maxr
        self.k['clear'](((max(self.n,nq,self.nc,self.nr)+127)//128,width),(128,),
            (w['out'],w['q'],w['y'],w['mq'],w['mp'],w['fc'],w['count'],self.n,nq,self.nc,self.nr,width,meta))
        w['tc'].fill(0)
        nodes,count=(w['no_targets'],w['no_count']) if targets is None else targets
        full=int(targets is None)
        self.k['targets'](((self.n+127)//128,),(128,),
            (self.ip,self.ic,self.ir,self.ptr,nodes,count,w['tc'],w['ti'],self.n,full))
        self.k['source'](((self.n+127)//128,width),(128,),
            (self.ip,self.ic,self.ir,self.ptr,F,w['fc'],w['fi'],self.n,width,meta))
        self.k['up']((len(self.leaves),width),(128,),
            (self.leaves,len(self.leaves),self.parent,self.rank,self.bp,self.ep,self.B,self.T,self.ids,self.ptr,
             w['fc'],w['fi'],F,w['q'],self.maxr,width,meta))
        self.k['schedule'](((len(self.aa)+127)//128,width),(128,),
            (self.aa,self.bb,len(self.aa),w['fc'],w['tc'],w['count'],w['tasks'],width,meta))
        self.k['couple']((128,width),(128,),
            (self.aa,self.bb,self.kind,self.lr,self.vp,self.ptr,self.rank,w['fc'],w['tc'],w['fi'],w['ti'],self.ids,
             self.V,F,w['q'],w['y'],w['out'],self.maxr,w['count'],w['tasks'],width,meta))
        self.k['down']((len(self.leaves),width),(128,),
            (self.leaves,len(self.leaves),self.parent,self.rank,self.bp,self.ep,self.B,self.T,self.ids,self.ptr,
             w['tc'],w['ti'],w['y'],w['out'],self.maxr,width,meta))
        if modal_force is None:
            self.k['modal_up']((self.nr,width),(128,),
                (self.P,self.nr,w['fc'],w['fi'],self.ids,F,w['mq'],width,meta))
            mq=w['mq']
        else:mq=modal_force.reshape(self.nr,width)
        self.k['modal_matrix']((self.nr,width),(32,),
            (self.Kinv if modal_inverse is None else modal_inverse,mq,w['mp'],self.nr,width,meta))
        self.k['finish'](((self.n+3)//4,width),(128,),
            (nodes,count,full,F,self.d,self.P,w['mp'],self.nr,w['out'],self.n,width,meta))
        return w['out'][:,0] if one else w['out']

    def storage(self):
        work=sum(v.nbytes for w in self.work.values() for v in w.values())
        return dict(operator_parameter_bytes=self.parameter_bytes,query_workspace_bytes=work,
            stored_bytes=self.parameter_bytes+work,widths=list(self.work),dense_surface_or_region_matrix_bytes=0,
            online_stiffness_factor_actions=False,online_CPU_flexibility_refills=0)
