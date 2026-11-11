"""Contact graph reads current CSR buffers directly; no capacity-wide copies."""
import numpy as np
import cupy as cp
from region_working import RegionWorkingSet
from conditional_graph import NativeGraph
from working_graph import bucket

CODE=r'''
extern "C" __global__ void finish_contact(const long long*keys,const long long*ids,const double*w,const int*meta,
 double*lam,long long*oldkeys,double*oldw,int*oldcount,int m){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 // All lam rows are assigned in one launch, including separation/empty cases.
 if(i<m){double f=0.;for(int j=0;j<meta[2];j++)if(ids[j]==i)f=w[j];lam[i]=f;}
 if(i<meta[2]){oldkeys[i]=keys[ids[i]];oldw[i]=w[i];}if(!i)oldcount[0]=meta[2];
}
'''

class AliasedWorkingSet(RegionWorkingSet):
    def surface_size(self,B,C):
        return C.shape[0]

    def __init__(self,*args,**kw):
        super().__init__(*args,**kw)
        self.fused_finish=cp.RawKernel(CODE,'finish_contact')

    def allocate(self,m,nnz,tolerance):
        self.mcap=bucket(m);self.nnzcap=bucket(nnz);cap=self.cap
        self.operator_signature=tuple((int(C.data.ptr),C.shape,C.strides) for B,C,v in self.device.blocks)
        self.map_signature=tuple((B.indptr.data.ptr,B.indices.data.ptr,B.data.data.ptr,v.data.ptr) for B,C,v in self.device.blocks)
        self.tolerance=tolerance
        self.meta=cp.zeros(7,dtype=cp.int32);self.ids=cp.zeros(cap,dtype=cp.int64);self.weights=cp.zeros(cap)
        self.W=cp.zeros((self.mcap,cap));self.b=cp.zeros(self.mcap);self.C=cp.zeros((cap,cap));self.rhs=cp.zeros(cap)
        self.L=cp.empty((cap,cap));self.inner=cp.zeros(4,dtype=cp.int32);nb=(self.mcap+255)//256
        self.partial=cp.empty((nb,2));self.worst=cp.empty(nb,dtype=cp.int32)
        self.buffers=[(B.indptr,B.indices,B.data,v,C,cp.empty((self.surface_size(B,C),cap))) for B,C,v in self.device.blocks]
        self.region=[(cp.full(self.surface_size(B,C),-1,dtype=cp.int32),cp.empty(self.surface_size(B,C),dtype=cp.int32),cp.zeros(1,dtype=cp.int32)) for B,C,v in self.device.blocks]
        def expansion():
            for i,((ptr,col,val,v,C,T),(seen,nodes,count)) in enumerate(zip(self.buffers,self.region)):
                p=len(seen)
                self.region_k['region_response'](((p+127)//128,),(128,),(C,ptr,col,val,self.ids,self.meta,T,np.int32(p),np.int32(cap),np.int32(self.device.column_major),nodes,count))
                self.region_k['region_contact'](((self.mcap+127)//128,),(128,),(ptr,col,val,v,self.ids,self.meta,T,self.W,np.int32(cap),np.int32(i>0),seen))
            self.k['gather_block'](((cap*cap+255)//256,),(256,),(self.W,self.b,self.ids,self.meta,self.C,self.rhs,self.weights,np.int32(cap)))
            self.correct((1,),(128,),(self.C,self.rhs,self.weights,self.weights,self.inner,self.meta,self.L,np.int32(cap),np.float64(tolerance),np.int32(1000),np.int32(self.local_sweeps)),shared_mem=(6*cap+128)*8+2*cap*4)
            self.k['check_all']((nb,),(256,),(self.W,self.b,self.weights,self.ids,self.meta,self.partial,self.worst,np.int32(cap)))
            self.k['next_column']((1,),(32,),(self.partial,self.worst,self.inner,self.meta,self.ids,self.weights,np.int32(nb),np.int32(cap),np.float64(tolerance)))
        self.meta[4]=1;expansion();cp.cuda.get_current_stream().synchronize()
        self.body_graph=NativeGraph.capture(expansion);self.loop=self.body_graph.while_loop(self.meta,limit=max(16,4*m))
        self.graph=self.body_graph;self.valid=cp.ones(m,dtype=cp.int32);self.lam=cp.zeros(m);self.rebuilds+=1

    def ensure(self,m,tolerance):
        signature=tuple((B.indptr.data.ptr,B.indices.data.ptr,B.data.data.ptr,v.data.ptr) for B,C,v in self.device.blocks)
        if signature!=getattr(self,'map_signature',None):self.graph=None
        return super().ensure(m,tolerance)

    def enqueue_prepare(self,b,keys,valid=None):
        m=len(b);self.b[:m]=b
        # device_start assigns every weight used by meta[2]; finish assigns lam.
        for (ptr,col,val,rank,C,T),(seen,nodes,count) in zip(self.buffers,self.region):
            seen.fill(-1);count.fill(0)
            self.region_k['region_nodes'](((len(val)+127)//128,),(128,),(col,val,seen,nodes,count,np.int32(len(val)),ptr,np.int32(m)))
        self.region_k['device_start']((1,),(1,),(self.b,keys,self.oldkeys,self.oldw,self.oldcount,self.valid if valid is None else valid,self.meta,self.ids,self.weights,np.int32(m),np.int32(self.cap),np.int32(self.reuse_history)))

    def enqueue_finish(self,keys):
        m=len(self.lam);n=max(m,self.cap)
        self.fused_finish(((n+127)//128,),(128,),(keys,self.ids,self.weights,self.meta,self.lam,self.oldkeys,self.oldw,self.oldcount,np.int32(m)))
        return self.lam
