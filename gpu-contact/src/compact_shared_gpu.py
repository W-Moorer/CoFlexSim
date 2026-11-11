"""GPU-only sparse source/target scheduling for shared compliance actions.

No host decision or matrix entry download in a query. Every cross-region block
with a nonzero source and requested target is kept. This is an operator query,
not yet a replacement for the time integrator's dense working-region API.
"""
import numpy as np
import cupy as cp
from global_local_compliance import dofs
from shared_compliance_gpu import SharedGPUAction, CODE as OLD_CODE

CODE=r'''
extern "C" __global__ void scatter_source(const int*ip,const int*ic,const int*ir,
 const int*ptr,int n,const double*f,int*fc,int*fi){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=n||f[g]==0.)return;
 for(int k=ip[g];k<ip[g+1];k++){int c=ic[k];fi[ptr[c]+atomicAdd(fc+c,1)]=ir[k];}}
extern "C" __global__ void scatter_target(const int*ip,const int*ic,const int*ir,
 const int*ptr,const int*t,int nt,int*tc,int*ti){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=nt)return;int g=t[j];
 for(int k=ip[g];k<ip[g+1];k++){int c=ic[k];ti[ptr[c]+atomicAdd(tc+c,1)]=ir[k];}}
extern "C" __global__ void schedule(const int*aa,const int*bb,int nb,const int*fc,const int*tc,int*count,int*tasks){
 int z=blockIdx.x*blockDim.x+threadIdx.x;if(z>=nb)return;int a=aa[z],b=bb[z];
 if(fc[b]&&tc[a])tasks[atomicAdd(count,1)]=2*z;
 if(a!=b&&fc[a]&&tc[b])tasks[atomicAdd(count,1)]=2*z+1;}
extern "C" __global__ void compact_couple(const int*aa,const int*bb,const int*kind,const int*lr,const long long*vp,
 const int*ptr,const int*rank,const int*fc,const int*tc,const int*fi,const int*ti,const int*ids,
 const double*V,const double*f,const double*q,double*y,double*out,int maxr,const int*count,const int*tasks){
 __shared__ double tmp[128];
 for(int task=blockIdx.x;task<count[0];task+=gridDim.x){
  int id=tasks[task],z=id/2,rev=id%2,a=aa[z],b=bb[z],dest=rev?b:a,src=rev?a:b;
  int na=ptr[a+1]-ptr[a],nb=ptr[b+1]-ptr[b];
  if(kind[z]==1){
   int r=threadIdx.x;if(r<rank[dest]){double v=0.;
    for(int j=0;j<rank[src];j++)v+=V[vp[z]+(rev?(long long)j*rank[b]+r:(long long)r*rank[b]+j)]*q[src*maxr+j];
    atomicAdd(y+dest*maxr+r,v);}
  }else if(kind[z]==2){
   int rankn=lr[z],r=threadIdx.x;long long sb=vp[z]+(rev?0:(long long)na*rankn);
   if(r<rankn){double v=0.;
    for(int k=0;k<fc[src];k++){int j=fi[ptr[src]+k];v+=V[sb+(long long)j*rankn+r]*f[ids[ptr[src]+j]];}tmp[r]=v;}
   __syncthreads();long long db=vp[z]+(rev?(long long)na*rankn:0);
   for(int k=threadIdx.x;k<tc[dest];k+=blockDim.x){int j=ti[ptr[dest]+k];double v=0.;
    for(int r=0;r<rankn;r++)v+=V[db+(long long)j*rankn+r]*tmp[r];atomicAdd(out+ids[ptr[dest]+j],v);}
  }else{
   for(int k=threadIdx.x;k<tc[dest];k+=blockDim.x){int r=ti[ptr[dest]+k];double v=0.;
    for(int l=0;l<fc[src];l++){int j=fi[ptr[src]+l];v+=V[vp[z]+(rev?(long long)j*nb+r:(long long)r*nb+j)]*f[ids[ptr[src]+j]];}
    atomicAdd(out+ids[ptr[dest]+r],v);}
  }
  __syncthreads();
 }}
'''


class CompactSharedGPUAction(SharedGPUAction):
    def __init__(self,model,force,targets,protect=True):
        super().__init__(model,force,targets,protect)
        # Inverse incidence: only loaded/requested dofs visit their ancestors.
        ids=np.concatenate([dofs(c.nodes) for c in model.clusters])
        ptr=np.r_[0,np.cumsum([3*len(c.nodes) for c in model.clusters])]
        owner=np.repeat(np.arange(self.nc,dtype=np.int32),np.diff(ptr))
        row=np.arange(len(ids))-ptr[owner];order=np.argsort(ids,kind='stable')
        ip=np.r_[0,np.cumsum(np.bincount(ids,minlength=self.n))].astype(np.int32)
        self.ip=cp.asarray(ip);self.ic=cp.asarray(owner[order]);self.ir=cp.asarray(row[order],dtype=cp.int32)
        self.kind=cp.asarray([{'exact':0,'far':1,'local':2}[k] for a,b,k,v in model.blocks],dtype=cp.int32)
        self.lr=cp.asarray([v.shape[1] if k=='local' else 0 for a,b,k,v in model.blocks],dtype=cp.int32)
        if any(k=='local' and v.shape[1]>128 for a,b,k,v in model.blocks):raise ValueError('Local rank exceeds kernel capacity')
        self.count=cp.zeros(1,dtype=cp.int32);self.tasks=cp.empty(2*len(self.aa),dtype=cp.int32)
        for name in ['scatter_source','scatter_target','schedule','compact_couple']:self.k[name]=cp.RawKernel(CODE,name)
        self.parameter_bytes+=sum(a.nbytes for a in [self.ip,self.ic,self.ir,self.lr])
        self.workspace_bytes+=self.count.nbytes+self.tasks.nbytes

    def __call__(self):
        self.fc.fill(0);self.tc.fill(0);self.q.fill(0);self.y.fill(0);self.out.fill(0);self.count.fill(0)
        self.k['scatter_source'](((self.n+127)//128,),(128,),
            (self.ip,self.ic,self.ir,self.ptr,self.n,self.f,self.fc,self.fi))
        self.k['scatter_target'](((len(self.targets)+127)//128,),(128,),
            (self.ip,self.ic,self.ir,self.ptr,self.targets,len(self.targets),self.tc,self.ti))
        self.k['schedule'](((len(self.aa)+127)//128,),(128,),
            (self.aa,self.bb,len(self.aa),self.fc,self.tc,self.count,self.tasks))
        self.k['project']((self.nc,),(128,),(self.ids,self.ptr,self.rank,self.bp,self.fc,self.fi,self.B,self.f,self.q,self.maxr))
        self.k['compact_couple']((128,),(128,),(self.aa,self.bb,self.kind,self.lr,self.vp,self.ptr,self.rank,
            self.fc,self.tc,self.fi,self.ti,self.ids,self.V,self.f,self.q,self.y,self.out,self.maxr,self.count,self.tasks))
        self.k['restore']((self.nc,),(128,),(self.ids,self.ptr,self.rank,self.bp,self.tc,self.ti,self.B,self.y,self.out,self.maxr))
        r=len(self.mq)
        if r:self.k['modal_project'](((r+127)//128,),(128,),(self.modal,self.f,self.fc,self.fi,self.ids,self.mq,r))
        self.k['modal_restore'](((len(self.targets)+127)//128,),(128,),
            (self.modal,self.mq,self.f,self.d,self.targets,len(self.targets),self.out,r,self.sign,int(self.protect)))
        return self.out[self.targets]
