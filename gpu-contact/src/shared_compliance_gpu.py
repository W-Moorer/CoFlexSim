"""Resident operator-action prototype of shared_hierarchical_compliance.

One kernel family for all spatial partitions, no gear-specific dispatch. The
query is a supplied nodal force -> supplied target-node displacement action,
not construction of a dense R_II matrix and not a full contact time step.
"""
import numpy as np
import cupy as cp
from global_local_compliance import dofs

CODE=r'''
extern "C" __global__ void active(const int*ids,const int*owners,const int*ptr,int n,
 const double*f,const int*mask,int*fc,int*tc,int*fi,int*ti){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=n)return;int c=owners[k],r=k-ptr[c],g=ids[k];
 if(f[g]!=0.)fi[ptr[c]+atomicAdd(fc+c,1)]=r;
 if(mask[g])ti[ptr[c]+atomicAdd(tc+c,1)]=r;}
extern "C" __global__ void project(const int*ids,const int*ptr,const int*rank,const long long*bp,
 const int*fc,const int*fi,const double*B,const double*f,double*q,int maxr){
 int c=blockIdx.x,r=threadIdx.x;if(r>=rank[c])return;double v=0.;
 for(int k=0;k<fc[c];k++){int j=fi[ptr[c]+k];v+=B[bp[c]+(long long)j*rank[c]+r]*f[ids[ptr[c]+j]];}
 q[c*maxr+r]=v;}
extern "C" __global__ void couple(const int*aa,const int*bb,const int*kind,const long long*vp,
 const int*ptr,const int*rank,const int*fc,const int*tc,const int*fi,const int*ti,const int*ids,
 const double*V,const double*f,const double*q,double*y,double*out,int maxr){
 int z=blockIdx.x/2,rev=blockIdx.x%2,a=aa[z],b=bb[z],dest=rev?b:a,src=rev?a:b;
 if(rev&&a==b)return;if(!fc[src])return;
 if(kind[z]){
  int r=threadIdx.x;if(r>=rank[dest])return;double v=0.;
  for(int j=0;j<rank[src];j++)v+=V[vp[z]+(rev?(long long)j*rank[b]+r:(long long)r*rank[b]+j)]*q[src*maxr+j];
  atomicAdd(y+dest*maxr+r,v);
 }else{
  int nb=ptr[b+1]-ptr[b];
  for(int k=threadIdx.x;k<tc[dest];k+=blockDim.x){
   int r=ti[ptr[dest]+k];double v=0.;
   for(int l=0;l<fc[src];l++){int j=fi[ptr[src]+l];v+=V[vp[z]+(rev?(long long)j*nb+r:(long long)r*nb+j)]*f[ids[ptr[src]+j]];}
   atomicAdd(out+ids[ptr[dest]+r],v);
  }
 }}
extern "C" __global__ void restore(const int*ids,const int*ptr,const int*rank,const long long*bp,
 const int*tc,const int*ti,const double*B,const double*y,double*out,int maxr){
 int c=blockIdx.x;if(!rank[c])return;
 for(int k=threadIdx.x;k<tc[c];k+=blockDim.x){int r=ti[ptr[c]+k];double v=0.;
 for(int j=0;j<rank[c];j++)v+=B[bp[c]+(long long)r*rank[c]+j]*y[c*maxr+j];
 atomicAdd(out+ids[ptr[c]+r],v);}}
extern "C" __global__ void modal_project(const double*B,const double*f,const int*fc,const int*fi,
 const int*ids,double*q,int rank){int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=rank)return;
 double v=0.;for(int k=0;k<fc[0];k++){int r=ids[fi[k]];v+=B[(long long)r*rank+j]*f[r];}q[j]=v;}
extern "C" __global__ void modal_restore(const double*B,const double*q,const double*f,const double*d,
 const int*targets,int nt,double*out,int rank,int sign,int protect){int k=blockIdx.x*blockDim.x+threadIdx.x;
 if(k>=nt)return;int r=targets[k];double v=0.;for(int j=0;j<rank;j++)v+=B[(long long)r*rank+j]*q[j];
 out[r]+=sign*v+(protect?d[r]*f[r]:0.);}
'''


class SharedGPUAction:
    def __init__(self,model,force,targets,protect=True):
        self.model=model;self.protect=protect;self.n=model.n;self.nc=len(model.clusters)
        self.maxr=max(1,max((0 if c.basis is None else c.basis.shape[1]) for c in model.clusters))
        ids=[dofs(c.nodes).astype(np.int32) for c in model.clusters]
        ptr=np.r_[0,np.cumsum([len(i) for i in ids])].astype(np.int32)
        ranks=np.array([0 if c.basis is None else c.basis.shape[1] for c in model.clusters],np.int32)
        bases=[c.basis.ravel() for c in model.clusters if c.basis is not None]
        bp=np.r_[0,np.cumsum([0 if c.basis is None else c.basis.size for c in model.clusters])].astype(np.int64)
        self.ids=cp.asarray(np.concatenate(ids));self.ptr=cp.asarray(ptr)
        self.owners=cp.asarray(np.repeat(np.arange(self.nc,dtype=np.int32),np.diff(ptr)))
        self.rank=cp.asarray(ranks);self.bp=cp.asarray(bp);self.B=cp.asarray(np.concatenate(bases) if bases else np.zeros(0))
        self.aa=cp.asarray([a for a,b,k,v in model.blocks],dtype=cp.int32)
        self.bb=cp.asarray([b for a,b,k,v in model.blocks],dtype=cp.int32)
        self.kind=cp.asarray([k=='far' for a,b,k,v in model.blocks],dtype=cp.int32)
        self.vp=cp.asarray(np.r_[0,np.cumsum([v.size for a,b,k,v in model.blocks])],dtype=cp.int64)
        self.V=cp.asarray(np.concatenate([v.ravel() for a,b,k,v in model.blocks]))
        self.f=cp.asarray(force);self.targets=cp.asarray(targets,dtype=cp.int32)
        self.mask=cp.zeros(self.n,dtype=cp.int32);self.mask[self.targets]=1
        self.fc=cp.zeros(self.nc,dtype=cp.int32);self.tc=cp.zeros_like(self.fc)
        self.fi=cp.empty(len(self.ids),dtype=cp.int32);self.ti=cp.empty_like(self.fi)
        self.q=cp.zeros((self.nc,self.maxr));self.y=cp.zeros_like(self.q);self.out=cp.zeros(self.n)
        self.d=cp.asarray(model.error_rows)
        factor=model.subtract if model.subtract is not None else model.U
        if factor is None:factor=np.zeros((self.n,0))
        self.modal=cp.asarray(factor);self.mq=cp.empty(factor.shape[1]);self.sign=-1 if model.subtract is not None else 1
        self.k={n:cp.RawKernel(CODE,n) for n in ['active','project','couple','restore','modal_project','modal_restore']}
        self.parameter_bytes=sum(a.nbytes for a in [self.ids,self.ptr,self.owners,self.rank,self.bp,self.B,
            self.aa,self.bb,self.kind,self.vp,self.V,self.d,self.modal])
        self.workspace_bytes=sum(a.nbytes for a in [self.f,self.targets,self.mask,self.fc,self.tc,self.fi,self.ti,self.q,self.y,self.out,self.mq])

    def __call__(self):
        self.fc.fill(0);self.tc.fill(0);self.q.fill(0);self.y.fill(0);self.out.fill(0)
        self.k['active'](((len(self.ids)+127)//128,),(128,),
            (self.ids,self.owners,self.ptr,len(self.ids),self.f,self.mask,self.fc,self.tc,self.fi,self.ti))
        self.k['project']((self.nc,),(128,),(self.ids,self.ptr,self.rank,self.bp,self.fc,self.fi,self.B,self.f,self.q,self.maxr))
        self.k['couple']((2*len(self.aa),),(128,),(self.aa,self.bb,self.kind,self.vp,self.ptr,self.rank,
            self.fc,self.tc,self.fi,self.ti,self.ids,self.V,self.f,self.q,self.y,self.out,self.maxr))
        self.k['restore']((self.nc,),(128,),(self.ids,self.ptr,self.rank,self.bp,self.tc,self.ti,self.B,self.y,self.out,self.maxr))
        r=len(self.mq)
        if r:self.k['modal_project'](((r+127)//128,),(128,),(self.modal,self.f,self.fc,self.fi,self.ids,self.mq,r))
        self.k['modal_restore'](((len(self.targets)+127)//128,),(128,),
            (self.modal,self.mq,self.f,self.d,self.targets,len(self.targets),self.out,r,self.sign,int(self.protect)))
        return self.out[self.targets]


def benchmark(model,F,reference,x):
    from scipy.spatial import cKDTree
    tree=cKDTree(x);records=[]
    # Highest resultant nodal-load norm plus late/end phases; no trajectory fit.
    indices=sorted(set([int(np.argmax(np.linalg.norm(F,axis=0))),9,39,69]))
    for j in indices:
        active=np.flatnonzero(np.any(F[:,j].reshape(-1,3)!=0,axis=1))
        targets=np.unique(tree.query(x[active],k=min(24,len(x)))[1].ravel())
        td=dofs(targets);query=SharedGPUAction(model,F[:,j],td)
        for _ in range(3):actual=query()
        cp.cuda.get_current_stream().synchronize()
        actual=cp.asnumpy(actual);cpu=model.action(F[:,j],True)[td]
        err=float(np.linalg.norm(actual-cpu)/max(np.linalg.norm(cpu),1e-300))
        if err>1e-9:raise AssertionError(('CPU GPU representation mismatch',err))
        stream=cp.cuda.Stream(non_blocking=True)
        with stream:stream.begin_capture();query();graph=stream.end_capture()
        times=[]
        for _ in range(3):
            a,b=cp.cuda.Event(),cp.cuda.Event();a.record()
            for k in range(100):graph.launch()
            b.record();b.synchronize();times.append(cp.cuda.get_elapsed_time(a,b)/100)
        individual=[]
        for _ in range(30):
            a,b=cp.cuda.Event(),cp.cuda.Event();a.record();graph.launch();b.record();b.synchronize()
            individual.append(cp.cuda.get_elapsed_time(a,b))
        records.append(dict(sample=j+1,force_nodes=len(active),target_nodes=len(targets),
            gpu_query_ms=times,median_ms=float(np.median(times)),max_batch_mean_ms=max(times),
            CPU_GPU_relative_error=err,parameters_bytes=query.parameter_bytes,workspace_bytes=query.workspace_bytes,
            individual_query_ms=individual,sampled_max_ms=max(individual),sampled_p95_ms=float(np.percentile(individual,95)),
            response_relative_to_exact=float(np.linalg.norm(actual-reference[td,j])/max(np.linalg.norm(reference[td,j]),1e-300)),
            scope='resident nodal force to targets (force nodes plus 24-neighbor support); all tree/coupling work included; excludes contact search/upload'))
        del query,graph;cp.get_default_memory_pool().free_all_blocks()
    return records
