"""Exact candidate-row response and device-controlled contact working set."""
from pathlib import Path
import cupy as cp
import numpy as np
from resident_working import ResidentWorkingSet
from conditional_graph import NativeGraph

CODE=r'''
extern "C" __global__ void region_nodes(const int*col,const double*value,int*seen,int*ids,int*count,int entries,const int*ptr,int rows){
 int e=blockIdx.x*blockDim.x+threadIdx.x;if(e<entries&&e<ptr[rows]&&value[e]!=0.){
  int c=col[e];if(atomicCAS(seen+c,-1,-2)==-1){int i=atomicAdd(count,1);ids[i]=c;seen[c]=i;}
 }
}
extern "C" __global__ void region_response(const double*C,const int*ptr,const int*col,const double*val,
 const long long*rows,const int*meta,double*T,int p,int cap,int column_major,const int*nodes,const int*count){
 int ii=blockIdx.x*blockDim.x+threadIdx.x;if(meta[4]||meta[5]||ii>=count[0])return;int i=nodes[ii];
 for(int j=meta[3];j<meta[2];j++){int row=rows[j];double v=0.;
  for(int e=ptr[row];e<ptr[row+1];e++)if(val[e]!=0.)v+=C[column_major?(long long)col[e]*p+i:(long long)i*p+col[e]]*val[e];
  T[ii*cap+j]=v;
 }
}
extern "C" __global__ void region_contact(const int*ptr,const int*col,const double*val,const double*v,
 const long long*rows,const int*meta,const double*T,double*W,int cap,int add,const int*seen){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(meta[4]||meta[5]||i>=meta[0])return;
 for(int j=meta[3];j<meta[2];j++){double s=v[i]*v[rows[j]];
  for(int e=ptr[i];e<ptr[i+1];e++)if(val[e]!=0.)s+=val[e]*T[seen[col[e]]*cap+j];
  if(add)W[i*cap+j]+=s;else W[i*cap+j]=s;
 }
}
extern "C" __global__ void device_start(const double*b,const long long*keys,const long long*oldkeys,
 const double*oldw,const int*oldcount,const int*valid,int*meta,long long*ids,double*weights,int m,int cap,int reuse){
 if(threadIdx.x)return;for(int i=0;i<7;i++)meta[i]=0;meta[0]=m;
 int j=0;for(int i=1;i<m;i++)if(b[i]<b[j])j=i;
 if(!m||b[j]>=0.){meta[4]=1;return;}
 int k=0;if(reuse)for(int i=0;i<m;i++)if(valid[i])for(int z=0;z<oldcount[0];z++)if(keys[i]==oldkeys[z]&&oldw[z]>1e-8){
  if(k==cap){meta[5]=5;return;}ids[k]=i;weights[k++]=oldw[z];break;
 }
 meta[1]=k;bool found=false;for(int i=0;i<k;i++)found=found||ids[i]==j;
 if(!found){if(k==cap){meta[5]=5;return;}ids[k]=j;weights[k++]=0.;}meta[2]=k;
}
'''

class RegionWorkingSet(ResidentWorkingSet):
    def __init__(self,device,cap=32,**kw):
        super().__init__(device,cap=cap,**kw)
        mod=cp.RawModule(code=CODE)
        self.region_k={n:mod.get_function(n) for n in ('region_nodes','region_response','region_contact','device_start')}
        # The frozen corrector also solves padded identity rows up to capacity.
        # Those rows have positive free gaps and zero coupling. Remove only that
        # inactive padding; retain the complete physical working set and checks.
        source=Path(__file__).with_name('parallel_qp.cu').read_text()
        source=source.replace('int n,double tol,int maxit,int local_sweeps){','int stride,double tol,int maxit,int local_sweeps){\n int n=meta[2];')
        source=source.replace('C[i*n+i]','C[i*stride+i]').replace(',n)',',stride)')
        self.correct=cp.RawKernel(source,'parallel_qp')
        self.region_counts=[]

    def allocate(self,m,nnz,tolerance):
        super().allocate(m,nnz,tolerance)
        cap=self.cap;nb=(self.mcap+255)//256
        self.region=[]
        for _,_,_,_,C,_ in self.buffers:
            p=C.shape[0];self.region.append((cp.full(p,-1,dtype=cp.int32),cp.empty(p,dtype=cp.int32),cp.zeros(1,dtype=cp.int32)))
        def expansion():
            for i,((ptr,col,val,v,C,T),(seen,nodes,count)) in enumerate(zip(self.buffers,self.region)):
                p=C.shape[0]
                self.region_k['region_response'](((p+127)//128,),(128,),(C,ptr,col,val,self.ids,self.meta,T,np.int32(p),np.int32(cap),np.int32(self.device.column_major),nodes,count))
                self.region_k['region_contact'](((self.mcap+127)//128,),(128,),(ptr,col,val,v,self.ids,self.meta,T,self.W,np.int32(cap),np.int32(i>0),seen))
            self.k['gather_block'](((cap*cap+255)//256,),(256,),(self.W,self.b,self.ids,self.meta,self.C,self.rhs,self.weights,np.int32(cap)))
            self.correct((1,),(128,),(self.C,self.rhs,self.weights,self.weights,self.inner,self.meta,self.L,np.int32(cap),np.float64(tolerance),np.int32(1000),np.int32(self.local_sweeps)),shared_mem=(6*cap+128)*8+2*cap*4)
            self.k['check_all']((nb,),(256,),(self.W,self.b,self.weights,self.ids,self.meta,self.partial,self.worst,np.int32(cap)))
            self.k['next_column']((1,),(32,),(self.partial,self.worst,self.inner,self.meta,self.ids,self.weights,np.int32(nb),np.int32(cap),np.float64(tolerance)))
        self.meta[4]=1;expansion();cp.cuda.get_current_stream().synchronize()
        self.body_graph=NativeGraph.capture(expansion);self.loop=self.body_graph.while_loop(self.meta,limit=max(16,4*m))
        self.valid=cp.ones(m,dtype=cp.int32);self.lam=cp.zeros(m)

    def ensure(self,m,tolerance):
        nnz=max(len(B.data) for B,C,v in self.device.blocks)
        signature=tuple((int(C.data.ptr),C.shape,C.strides) for B,C,v in self.device.blocks)
        if self.graph is None or m>self.mcap or nnz>self.nnzcap or signature!=self.operator_signature or tolerance!=self.tolerance or len(self.lam)!=m:self.allocate(m,nnz,tolerance)

    def enqueue_prepare(self,b,keys,valid=None):
        m=len(b);self.b[:m]=b;self.weights.fill(0);self.lam.fill(0)
        for (B,C,v),(ptr,col,val,rank,_,T),(seen,nodes,count) in zip(self.device.blocks,self.buffers,self.region):
            ptr[:m+1]=B.indptr[:m+1];col[:len(B.data)]=B.indices;val[:len(B.data)]=B.data;rank[:m]=v[:m]
            seen.fill(-1);count.fill(0)
            self.region_k['region_nodes'](((len(B.data)+127)//128,),(128,),(col,val,seen,nodes,count,np.int32(len(B.data)),ptr,np.int32(m)))
        self.region_k['device_start']((1,),(1,),(self.b,keys,self.oldkeys,self.oldw,self.oldcount,self.valid if valid is None else valid,self.meta,self.ids,self.weights,np.int32(m),np.int32(self.cap),np.int32(self.reuse_history)))
    def enqueue_finish(self,keys):
        self.k['scatter_result']((1,),(self.cap,),(self.ids,self.weights,self.meta,self.lam,np.int32(self.cap)))
        self.remember((1,),(self.cap,),(keys,self.ids,self.weights,self.meta,self.oldkeys,self.oldw,self.oldcount,np.int32(self.cap)))
        return self.lam

    def enqueue(self,b,keys,valid=None):
        self.enqueue_prepare(b,keys,valid);self.loop.launch();return self.enqueue_finish(keys)

    def solve_current(self,b,keys,tolerance=1e-14):
        self.calls+=1
        if not len(b):return cp.zeros(0),dict(outer_iterations=0,stored_columns=0)
        self.ensure(len(b),tolerance);lam=self.enqueue(b,keys)
        state=cp.asnumpy(self.meta);self.host_reads+=1
        if state[5]:
            if state[5]==5 and self.cap<256:
                previous=self.cap;self.cap*=2;self.graph=None;self.capacity_growths+=1
                self.oldkeys=cp.pad(self.oldkeys,(0,self.cap-previous));self.oldw=cp.pad(self.oldw,(0,self.cap-previous))
                return self.solve_current(b,keys,tolerance)
            self.oldcount.fill(0)
            return self.fallback(b,(cp.empty(0,dtype=cp.int64),cp.empty(0)),tolerance,int(state[5]))
        self.device.statistics['device_response_columns']+=int(state[2]);self.expansions+=int(state[6]);self.temporal_seeds+=int(state[1])
        return lam.copy(),dict(outer_iterations=int(state[6]),stored_columns=int(state[2]))
