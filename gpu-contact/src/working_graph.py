"""Reusable capacity-bounded CUDA graph for unknown contact working sets.

Rank defects, capacity overflow or failed certificates fall back to the existing
GPU eigensolver path. No contact operator regularization or CPU numerical solve.
"""
from pathlib import Path
import numpy as np
import cupy as cp
from device_qp import working_set

def bucket(n):return 1<<max(0,int(n-1).bit_length())

class GraphWorkingSet:
    def __init__(self,device,cap=32):
        self.device=device;self.cap=cap;self.graph=None;self.mcap=0;self.nnzcap=0
        code=Path(__file__).with_suffix('.cu').read_text()
        if getattr(device,'column_major',False):code=code.replace('C[(long long)i*p+col[e]]','C[(long long)col[e]*p+i]')
        module=cp.RawModule(code=code)
        names=['set_start','new_response','new_columns','gather_block','check_all','next_column','scatter_result']
        self.k={n:module.get_function(n) for n in names}
        self.small=cp.RawKernel(Path(__file__).with_name('small_qp.cu').read_text(),'small_qp')
        self.fallbacks=0;self.calls=0;self.rebuilds=0

    def allocate(self,m,nnz,tolerance):
        self.mcap=bucket(m);self.nnzcap=bucket(nnz);cap=self.cap
        self.operator_signature=tuple((int(C.data.ptr),C.shape,C.strides) for B,C,v in self.device.blocks)
        self.tolerance=tolerance
        self.meta=cp.zeros(7,dtype=cp.int32);self.ids=cp.zeros(cap,dtype=cp.int64);self.weights=cp.zeros(cap)
        self.W=cp.zeros((self.mcap,cap));self.b=cp.zeros(self.mcap);self.C=cp.zeros((cap,cap));self.rhs=cp.zeros(cap)
        self.inner=cp.zeros(3,dtype=cp.int32);nb=(self.mcap+255)//256
        self.partial=cp.empty((nb,2));self.worst=cp.empty(nb,dtype=cp.int32);self.buffers=[]
        for B,C,v in self.device.blocks:
            self.buffers.append((cp.zeros(self.mcap+1,dtype=cp.int32),cp.zeros(self.nnzcap,dtype=cp.int32),
                cp.zeros(self.nnzcap),cp.zeros(self.mcap),C,cp.zeros((C.shape[0],cap))))
        def dispatch():
            for i,(ptr,col,val,v,C,T) in enumerate(self.buffers):
                p=C.shape[0]
                self.k['new_response'](((p*cap+255)//256,),(256,),(C,ptr,col,val,self.ids,self.meta,T,np.int32(p),np.int32(cap)))
                self.k['new_columns'](((self.mcap*cap+255)//256,),(256,),(ptr,col,val,v,self.ids,self.meta,T,self.W,np.int32(cap),np.int32(i>0)))
            self.k['gather_block'](((cap*cap+255)//256,),(256,),(self.W,self.b,self.ids,self.meta,self.C,self.rhs,self.weights,np.int32(cap)))
            self.small((1,),(32,),(self.C,self.rhs,self.weights,self.weights,self.inner,np.int32(cap),np.float64(tolerance),np.int32(1000)),shared_mem=(cap*cap+6*cap)*8+2*cap*4)
            self.k['check_all']((nb,),(256,),(self.W,self.b,self.weights,self.ids,self.meta,self.partial,self.worst,np.int32(cap)))
            self.k['next_column']((1,),(32,),(self.partial,self.worst,self.inner,self.meta,self.ids,self.weights,np.int32(nb),np.int32(cap),np.float64(tolerance)))
        dispatch();cp.cuda.get_current_stream().synchronize()
        stream=cp.cuda.Stream(non_blocking=True)
        with stream:
            stream.begin_capture();dispatch();self.graph=stream.end_capture()
        self.rebuilds+=1

    def solve(self,b,seeds=None,tolerance=1e-14):
        self.calls+=1;m=len(b)
        if not m:return cp.zeros(0),dict(outer_iterations=0,stored_columns=0,graph=True)
        nnz=max(len(B.data) for B,C,v in self.device.blocks)
        if not isinstance(seeds,tuple):
            seeds=seeds or {};seeds=(cp.asarray(list(seeds),dtype=cp.int64),cp.asarray(list(seeds.values())))
        if len(seeds[0])>=self.cap:return self.fallback(b,seeds,tolerance)
        signature=tuple((int(C.data.ptr),C.shape,C.strides) for B,C,v in self.device.blocks)
        if (self.graph is None or m>self.mcap or nnz>self.nnzcap or
            signature!=self.operator_signature or tolerance!=self.tolerance):self.allocate(m,nnz,tolerance)
        self.b[:m]=b;self.weights.fill(0);self.ids[:len(seeds[0])]=seeds[0];self.weights[:len(seeds[1])]=seeds[1]
        self.meta.set(np.asarray([m,len(seeds[0]),0,0,0,0,0],dtype=np.int32))
        for (B,C,v),(ptr,col,val,rank,_,T) in zip(self.device.blocks,self.buffers):
            ptr[:m+1]=B.indptr;col[:len(B.data)]=B.indices;val[:len(B.data)]=B.data;rank[:m]=v
        self.k['set_start']((1,),(32,),(self.b,self.meta,self.ids,self.weights,np.int32(self.cap)))
        for _ in range(m):
            self.graph.launch();state=cp.asnumpy(self.meta)
            if state[5]:return self.fallback(b,seeds,tolerance)
            if state[4]:
                lam=cp.zeros(m)
                self.k['scatter_result']((1,),(self.cap,),(self.ids,self.weights,self.meta,lam,np.int32(self.cap)))
                self.device.statistics['device_response_columns']+=int(state[2])
                return lam,dict(outer_iterations=int(state[6]),stored_columns=int(state[2]),graph=True)
        raise RuntimeError('Graph contact working set unresolved')

    def fallback(self,b,seeds,tolerance):
        self.fallbacks+=1
        return working_set(self.device.columns,b,seeds,tolerance)
