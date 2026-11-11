"""Entire-domain resident nested action with device-selected changing queries.

No bank callbacks, CPU refill, regional dense matrix or expanded internal basis.
The batch table is a test input source; the operator/workspace accepts any listed
surface indices. All switching and source/target list construction is on GPU.
"""
import numpy as np
import cupy as cp
from global_local_compliance import dofs
from compact_shared_gpu import CODE as COUPLE_CODE

CODE=r'''
extern "C" __global__ void clear_work(double*f,double*out,double*q,double*y,double*mq,double*mp,
 int*fc,int*tc,int*count,int n,int nq,int nc,int mr){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n){f[i]=0.;out[i]=0.;}if(i<nq){q[i]=0.;y[i]=0.;}
 if(i<mr){mq[i]=0.;mp[i]=0.;}if(i<nc){fc[i]=0;tc[i]=0;}if(i==0)*count=0;
}
extern "C" __global__ void choose_inputs(const int*caseid,const int*sp,const int*si,const double*sv,
 const int*tp,const int*tin,const int*ip,const int*ic,const int*ir,const int*ptr,
 double*f,int*fc,int*tc,int*fi,int*ti,int*targets,int*nt){
 int i=blockIdx.x*blockDim.x+threadIdx.x,c=*caseid;
 if(i==0)*nt=tp[c+1]-tp[c];
 if(i<sp[c+1]-sp[c]){int g=si[sp[c]+i];double v=sv[sp[c]+i];f[g]=v;
  if(v!=0.)for(int j=ip[g];j<ip[g+1];j++){int a=ic[j];fi[ptr[a]+atomicAdd(fc+a,1)]=ir[j];}}
 if(i<tp[c+1]-tp[c]){int g=tin[tp[c]+i];targets[i]=g;
  for(int j=ip[g];j<ip[g+1];j++){int a=ic[j];ti[ptr[a]+atomicAdd(tc+a,1)]=ir[j];}}
}
extern "C" __global__ void up(const int*leaves,int nl,const int*parent,const int*rank,const long long*bp,
 const long long*ep,const double*B,const double*T,const int*ids,const int*ptr,const int*fc,const int*fi,
 const double*f,double*q,int maxr){
 __shared__ double v[128],w[128];int l=blockIdx.x,r=threadIdx.x;if(l>=nl)return;
 int c=leaves[l];if(!fc[c]||!rank[c])return;
 double a=0.;if(r<rank[c]){for(int k=0;k<fc[c];k++){int j=fi[ptr[c]+k];a+=B[bp[c]+(long long)j*rank[c]+r]*f[ids[ptr[c]+j]];}
 v[r]=a;atomicAdd(q+c*maxr+r,a);}__syncthreads();
 while(parent[c]>=0){int p=parent[c];if(!rank[p])break;
  a=0.;if(r<rank[p]){for(int j=0;j<rank[c];j++)a+=T[ep[c]+(long long)j*rank[p]+r]*v[j];w[r]=a;atomicAdd(q+p*maxr+r,a);}
  __syncthreads();if(r<rank[p])v[r]=w[r];__syncthreads();c=p;
 }
}
extern "C" __global__ void modal_up(const double*P,int mr,const int*fc,const int*fi,const int*ids,
 const double*f,double*q){
 __shared__ double v[128];int lane=threadIdx.x%32,warp=threadIdx.x/32,r=blockIdx.x*32+lane;
 double a=0.;if(r<mr)for(int k=blockIdx.y*32+warp;k<min(fc[0],(int)(blockIdx.y+1)*32);k+=4){int g=ids[fi[k]];a+=P[(long long)g*mr+r]*f[g];}
 v[threadIdx.x]=a;__syncthreads();if(warp==0&&r<mr)atomicAdd(q+r,v[lane]+v[32+lane]+v[64+lane]+v[96+lane]);
}
extern "C" __global__ void modal_solve(const double*K,const double*q,double*p,int n){
 int r=blockIdx.x,l=threadIdx.x;double a=0.;for(int j=l;j<n;j+=32)a+=K[(long long)r*n+j]*q[j];
 for(int d=16;d;d/=2)a+=__shfl_down_sync(0xffffffff,a,d);if(!l)p[r]=a;
}
extern "C" __global__ void down(const int*leaves,int nl,const int*parent,const int*rank,const long long*bp,
 const long long*ep,const double*B,const double*T,const int*ids,const int*ptr,const int*tc,const int*ti,
 const double*y,double*out,int maxr){
 __shared__ double v[128],w[128];int l=blockIdx.x,r=threadIdx.x;if(l>=nl)return;
 int leaf=leaves[l];if(!tc[leaf]||!rank[leaf])return;
 int path[32],depth=0,c=leaf;while(c>=0&&depth<32){path[depth++]=c;c=parent[c];}
 for(int j=depth-1;j>=0;j--){c=path[j];int p=parent[c];double a=0.;
  if(r<rank[c]){a=y[c*maxr+r];if(p>=0&&rank[p])for(int k=0;k<rank[p];k++)a+=T[ep[c]+(long long)r*rank[p]+k]*v[k];w[r]=a;}
  __syncthreads();if(r<rank[c])v[r]=w[r];__syncthreads();
 }
 for(int k=r;k<tc[leaf];k+=blockDim.x){int j=ti[ptr[leaf]+k];double a=0.;for(int z=0;z<rank[leaf];z++)a+=B[bp[leaf]+(long long)j*rank[leaf]+z]*v[z];out[ids[ptr[leaf]+j]]+=a;}
}
extern "C" __global__ void finish(const int*targets,const int*nt,const double*out,const double*f,const double*d,
 const double*P,const double*mq,int mr,double*result,int*caseid,int cases){
 int k=blockIdx.x*4+threadIdx.x/32,l=threadIdx.x%32;
 if(k<*nt){int g=targets[k];double a=0.;for(int j=l;j<mr;j+=32)a+=P[(long long)g*mr+j]*mq[j];
 for(int s=16;s;s/=2)a+=__shfl_down_sync(0xffffffff,a,s);if(!l)result[k]=out[g]-a+d[g]*f[g];}
 if(blockIdx.x==0&&threadIdx.x==0)*caseid=(*caseid+1)%cases;
}
'''


class NestedResidentGPU:
    def __init__(self,model,P,Kinv,cases):
        self.n=model.n;self.nc=len(model.clusters);self.nr=P.shape[1]
        rank=np.array([0 if c.basis is None else c.basis.shape[1] for c in model.clusters],np.int32)
        self.maxr=max(1,int(rank.max()));assert self.maxr<=128
        rows=[dofs(c.nodes).astype(np.int32) for c in model.clusters]
        ids=np.concatenate(rows);ptr=np.r_[0,np.cumsum([len(x) for x in rows])].astype(np.int32)
        owner=np.repeat(np.arange(self.nc,dtype=np.int32),np.diff(ptr));row=np.arange(len(ids))-ptr[owner]
        order=np.argsort(ids,kind='stable');ip=np.r_[0,np.cumsum(np.bincount(ids,minlength=self.n))].astype(np.int32)
        leaves=np.array([c.id for c in model.clusters if c.children is None],np.int32)
        leafset=set(leaves);bases=[];bp=[];transfers=[];ep=[]
        bn=tn=0
        for c in model.clusters:
            bp.append(bn);ep.append(tn)
            if c.id in leafset and c.basis is not None:bases.append(c.basis.ravel());bn+=c.basis.size
            if c.id in model.transfers:transfers.append(model.transfers[c.id].ravel());tn+=model.transfers[c.id].size
        host=dict(ids=ids,ptr=ptr,ip=ip,ic=owner[order],ir=row[order].astype(np.int32),leaves=leaves,
            parent=np.array([model.parents.get(c.id,-1) for c in model.clusters],np.int32),rank=rank,
            bp=np.array(bp,np.int64),ep=np.array(ep,np.int64),B=np.concatenate(bases) if bases else np.empty(0),
            T=np.concatenate(transfers) if transfers else np.empty(0),
            aa=np.array([a for a,b,k,v in model.blocks],np.int32),bb=np.array([b for a,b,k,v in model.blocks],np.int32),
            kind=np.array([{'exact':0,'far':1,'local':2}[k] for a,b,k,v in model.blocks],np.int32),
            lr=np.array([v.shape[1] if k=='local' else 0 for a,b,k,v in model.blocks],np.int32),
            vp=np.r_[0,np.cumsum([v.size for a,b,k,v in model.blocks])].astype(np.int64),
            V=np.concatenate([v.ravel() for a,b,k,v in model.blocks]),d=model.error_rows,P=P,Kinv=Kinv)
        self.params={k:cp.asarray(v) for k,v in host.items()}
        self.parameter_bytes=sum(v.nbytes for v in self.params.values())
        for k,v in self.params.items():setattr(self,k,v)
        self.case_count=len(cases);self.maxs=max(len(c[0]) for c in cases);self.maxt=max(len(c[2]) for c in cases)
        self.inputs=dict(sp=cp.asarray(np.r_[0,np.cumsum([len(c[0]) for c in cases])],dtype=cp.int32),
            si=cp.asarray(np.concatenate([c[0] for c in cases]),dtype=cp.int32),
            sv=cp.asarray(np.concatenate([c[1] for c in cases]),dtype=cp.float64),
            tp=cp.asarray(np.r_[0,np.cumsum([len(c[2]) for c in cases])],dtype=cp.int32),
            tin=cp.asarray(np.concatenate([c[2] for c in cases]),dtype=cp.int32))
        self.work=dict(f=cp.zeros(self.n),out=cp.zeros(self.n),q=cp.zeros((self.nc,self.maxr)),y=cp.zeros((self.nc,self.maxr)),
            mq=cp.zeros(self.nr),mp=cp.zeros(self.nr),fc=cp.zeros(self.nc,dtype=cp.int32),tc=cp.zeros(self.nc,dtype=cp.int32),
            count=cp.zeros(1,dtype=cp.int32),tasks=cp.empty(2*len(self.aa),dtype=cp.int32),
            fi=cp.empty(len(ids),dtype=cp.int32),ti=cp.empty(len(ids),dtype=cp.int32),
            targets=cp.empty(self.maxt,dtype=cp.int32),nt=cp.zeros(1,dtype=cp.int32),
            result=cp.empty(self.maxt),caseid=cp.zeros(1,dtype=cp.int32))
        for k,v in self.work.items():setattr(self,k,v)
        self.workspace_bytes=sum(v.nbytes for v in self.work.values());self.input_bytes=sum(v.nbytes for v in self.inputs.values())
        self.k={name:cp.RawKernel(CODE,name) for name in ['clear_work','choose_inputs','up','modal_up','modal_solve','down','finish']}
        self.k['schedule']=cp.RawKernel(COUPLE_CODE,'schedule');self.k['couple']=cp.RawKernel(COUPLE_CODE,'compact_couple')

    def __call__(self):
        n=max(self.n,self.q.size,self.nc,self.nr)
        self.k['clear_work'](((n+127)//128,),(128,),(self.f,self.out,self.q,self.y,self.mq,self.mp,self.fc,self.tc,self.count,self.n,self.q.size,self.nc,self.nr))
        I=self.inputs
        self.k['choose_inputs'](((max(self.maxs,self.maxt)+127)//128,),(128,),
            (self.caseid,I['sp'],I['si'],I['sv'],I['tp'],I['tin'],self.ip,self.ic,self.ir,self.ptr,self.f,self.fc,self.tc,self.fi,self.ti,self.targets,self.nt))
        self.k['up']((len(self.leaves),),(128,),(self.leaves,len(self.leaves),self.parent,self.rank,self.bp,self.ep,self.B,self.T,self.ids,self.ptr,self.fc,self.fi,self.f,self.q,self.maxr))
        self.k['modal_up'](((self.nr+31)//32,max(1,(self.maxs+31)//32)),(128,),
            (self.P,self.nr,self.fc,self.fi,self.ids,self.f,self.mq))
        self.k['modal_solve']((self.nr,),(32,),(self.Kinv,self.mq,self.mp,self.nr))
        self.k['schedule'](((len(self.aa)+127)//128,),(128,),(self.aa,self.bb,len(self.aa),self.fc,self.tc,self.count,self.tasks))
        self.k['couple']((128,),(128,),(self.aa,self.bb,self.kind,self.lr,self.vp,self.ptr,self.rank,self.fc,self.tc,self.fi,self.ti,self.ids,self.V,self.f,self.q,self.y,self.out,self.maxr,self.count,self.tasks))
        self.k['down']((len(self.leaves),),(128,),(self.leaves,len(self.leaves),self.parent,self.rank,self.bp,self.ep,self.B,self.T,self.ids,self.ptr,self.tc,self.ti,self.y,self.out,self.maxr))
        self.k['finish'](((self.maxt+3)//4,),(128,),
            (self.targets,self.nt,self.out,self.f,self.d,self.P,self.mp,self.nr,self.result,self.caseid,self.case_count))
        return self.result
