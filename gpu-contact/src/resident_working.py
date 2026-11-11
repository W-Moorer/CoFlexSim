"""Device-resident temporal seeds and bounded batched working-set correction."""
from pathlib import Path
import numpy as np
import cupy as cp
from parallel_working import ParallelWorkingSet

CODE=r'''
extern "C" __global__ void resident_seed(const long long*keys,const long long*oldkeys,const double*oldw,const int*oldcount,
 const double*b,int*meta,long long*ids,double*weights,int m,int cap){
 if(threadIdx.x)return;int count=0;
 for(int i=0;i<m;i++)for(int j=0;j<oldcount[0];j++)if(keys[i]==oldkeys[j]&&oldw[j]>1e-8){
  if(count>=cap){meta[5]=5;return;}ids[count]=i;weights[count++]=oldw[j];break;
 }
 meta[1]=count;
}
extern "C" __global__ void remember(const long long*keys,const long long*ids,const double*weights,const int*meta,
 long long*oldkeys,double*oldw,int*count,int cap){
 int i=threadIdx.x;if(!i)count[0]=meta[2];if(i<meta[2]){oldkeys[i]=keys[ids[i]];oldw[i]=weights[i];}
}
extern "C" __global__ void next_batch(const double*partial,const int*worst,const int*inner,int*meta,
 long long*ids,double*weights,int blocks,int cap,double tolerance,const double*W,const double*b,int width){
 if(meta[4]||meta[5])return;int t=threadIdx.x;
 __shared__ int done,chosen;__shared__ double low[128];__shared__ int which[128];
 if(!t){
  meta[6]++;done=0;if(inner[0]){meta[5]=inner[0];done=1;}
  double minimum=1e300,ae=0.;for(int i=0;i<blocks;i++){minimum=fmin(minimum,partial[2*i]);ae=fmax(ae,partial[2*i+1]);}
  if(!done&&minimum>=-10*tolerance){if(ae>100*tolerance)meta[5]=6;else meta[4]=1;done=1;}
  if(!done){if(meta[2]>=cap){meta[5]=5;done=1;}else meta[3]=meta[2];}
 }
 __syncthreads();if(done)return;
 for(int slot=0;slot<width;slot++){
  double value=-10*tolerance;int candidate=-1;
  for(int r=t;r<meta[0];r+=blockDim.x){
   bool used=false;double gap=b[r];for(int j=0;j<meta[2];j++){used=used||ids[j]==r;if(j<meta[3])gap+=W[r*cap+j]*weights[j];}
   if(!used&&(gap<value||(gap==value&&(candidate<0||r<candidate)))){value=gap;candidate=r;}
  }
  low[t]=value;which[t]=candidate;__syncthreads();
  for(int s=64;s;s/=2){if(t<s&&(low[t+s]<low[t]||(low[t+s]==low[t]&&which[t+s]>=0&&(which[t]<0||which[t+s]<which[t])))){low[t]=low[t+s];which[t]=which[t+s];}__syncthreads();}
  if(!t){chosen=which[0];if(chosen>=0&&meta[2]<cap){int k=meta[2]++;ids[k]=chosen;weights[k]=0.;}else chosen=-1;}
  __syncthreads();if(chosen<0)break;
 }
}
'''

class ResidentWorkingSet(ParallelWorkingSet):
    def __init__(self,device,cap=32,batch=2,reuse_history=True,expansion_width=4):
        super().__init__(device,cap=cap,batch=batch)
        module=cp.RawModule(code=CODE);self.seed_kernel=module.get_function('resident_seed');self.remember=module.get_function('remember')
        self.oldkeys=cp.zeros(cap,dtype=cp.int64);self.oldw=cp.zeros(cap);self.oldcount=cp.zeros(1,dtype=cp.int32)
        self.temporal_seeds=0;self.capacity_growths=0;self.reuse_history=reuse_history;self.expansion_width=expansion_width
        self.next_batch_kernel=module.get_function('next_batch')
        self.k['next_column']=lambda grid,block,args:self.next_batch_kernel((1,),(128,),tuple(args)+(self.W,self.b,np.int32(self.expansion_width)))

    def solve_current(self,b,keys,tolerance=1e-14):
        self.calls+=1;m=len(b)
        if not m:return cp.zeros(0),dict(outer_iterations=0,stored_columns=0)
        nnz=max(len(B.data) for B,C,v in self.device.blocks)
        signature=tuple((int(C.data.ptr),C.shape,C.strides) for B,C,v in self.device.blocks)
        if self.graph is None or m>self.mcap or nnz>self.nnzcap or signature!=self.operator_signature or tolerance!=self.tolerance:self.allocate(m,nnz,tolerance)
        self.b[:m]=b;self.weights.fill(0);self.meta.set(np.asarray([m,0,0,0,0,0,0],dtype=np.int32))
        for (B,C,v),(ptr,col,val,rank,_,T) in zip(self.device.blocks,self.buffers):
            ptr[:m+1]=B.indptr[:m+1];col[:len(B.data)]=B.indices;val[:len(B.data)]=B.data;rank[:m]=v[:m]
        if self.reuse_history:self.seed_kernel((1,),(1,),(keys,self.oldkeys,self.oldw,self.oldcount,self.b,self.meta,self.ids,self.weights,np.int32(m),np.int32(self.cap)))
        self.k['set_start']((1,),(32,),(self.b,self.meta,self.ids,self.weights,np.int32(self.cap)))
        for _ in range((m+self.batch-1)//self.batch):
            self.graph.launch();state=cp.asnumpy(self.meta);self.host_reads+=1
            if state[5]:
                self.expansions+=int(state[6])
                if state[5]==5 and self.cap<256:
                    previous=self.cap;self.cap*=2;self.graph=None;self.capacity_growths+=1
                    self.oldkeys=cp.pad(self.oldkeys,(0,self.cap-previous));self.oldw=cp.pad(self.oldw,(0,self.cap-previous))
                    return self.solve_current(b,keys,tolerance)
                self.oldcount.fill(0)
                return self.fallback(b,(cp.empty(0,dtype=cp.int64),cp.empty(0)),tolerance,int(state[5]))
            if state[4]:
                lam=cp.zeros(m);self.k['scatter_result']((1,),(self.cap,),(self.ids,self.weights,self.meta,lam,np.int32(self.cap)))
                self.remember((1,),(self.cap,),(keys,self.ids,self.weights,self.meta,self.oldkeys,self.oldw,self.oldcount,np.int32(self.cap)))
                self.device.statistics['device_response_columns']+=int(state[2]);self.expansions+=int(state[6]);self.temporal_seeds+=int(state[1])
                return lam,dict(outer_iterations=int(state[6]),stored_columns=int(state[2]))
        raise RuntimeError('Resident contact workset unresolved')
