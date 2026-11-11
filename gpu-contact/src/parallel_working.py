"""Local contact prediction + coupled correction with batched device expansion.

Original graph and numerical baseline are unchanged. All response columns retain
whole-body cross coupling. Each graph expansion has a device completion guard;
several expansions execute without a host read. Exceptional dependent blocks
use the existing cuSOLVER-based rank-aware solver (never the serial small QP).
"""
from pathlib import Path
import numpy as np
import cupy as cp
from working_graph import bucket
from device_qp import lcp


class ParallelWorkingSet:
    def __init__(self, device, cap=64, batch=8, local_sweeps=2):
        self.device=device;self.cap=cap;self.batch=batch;self.local_sweeps=local_sweeps
        self.graph=None;self.mcap=0;self.nnzcap=0
        # Reuse the unchanged algebra kernels; completion guards make replay
        # idempotent after an accepted all-candidate certificate or a failure.
        code=Path(__file__).with_name('working_graph.cu').read_text()
        if getattr(device,'column_major',False):
            code=code.replace('C[(long long)i*p+col[e]]','C[(long long)col[e]*p+i]')
        for name in ['new_response','new_columns','gather_block','check_all','next_column']:
            start=code.index('void '+name+'(');brace=code.index('{',start)
            code=code[:brace+1]+'\n if(meta[4]||meta[5])return;'+code[brace+1:]
        module=cp.RawModule(code=code)
        names=['set_start','new_response','new_columns','gather_block','check_all','next_column','scatter_result']
        self.k={n:module.get_function(n) for n in names}
        cols=cp.RawModule(code=Path(__file__).with_name('parallel_columns.cu').read_text())
        self.k.update({n:cols.get_function(n) for n in ('response_rows','contact_rows')})
        self.correct=cp.RawKernel(Path(__file__).with_name('parallel_qp.cu').read_text(),'parallel_qp')
        self.calls=0;self.fallbacks=0;self.rebuilds=0;self.host_reads=0;self.expansions=0
        self.fallback_reasons={};self.local_predictor_sweeps=0

    def allocate(self,m,nnz,tolerance):
        self.mcap=bucket(m);self.nnzcap=bucket(nnz);cap=self.cap
        self.operator_signature=tuple((int(C.data.ptr),C.shape,C.strides) for B,C,v in self.device.blocks)
        self.tolerance=tolerance
        self.meta=cp.zeros(7,dtype=cp.int32);self.ids=cp.zeros(cap,dtype=cp.int64);self.weights=cp.zeros(cap)
        self.W=cp.zeros((self.mcap,cap));self.b=cp.zeros(self.mcap);self.C=cp.zeros((cap,cap));self.rhs=cp.zeros(cap)
        self.L=cp.empty((cap,cap));self.inner=cp.zeros(4,dtype=cp.int32);nb=(self.mcap+255)//256
        self.partial=cp.empty((nb,2));self.worst=cp.empty(nb,dtype=cp.int32);self.buffers=[]
        for B,C,v in self.device.blocks:
            self.buffers.append((cp.zeros(self.mcap+1,dtype=cp.int32),cp.zeros(self.nnzcap,dtype=cp.int32),
                                cp.zeros(self.nnzcap),cp.zeros(self.mcap),C,cp.zeros((C.shape[0],cap))))
        def expansion():
            for i,(ptr,col,val,v,C,T) in enumerate(self.buffers):
                p=C.shape[0]
                self.k['response_rows'](((p+127)//128,),(128,),(C,ptr,col,val,self.ids,self.meta,T,np.int32(p),np.int32(cap),np.int32(getattr(self.device,'column_major',False))))
                self.k['contact_rows'](((self.mcap+127)//128,),(128,),(ptr,col,val,v,self.ids,self.meta,T,self.W,np.int32(cap),np.int32(i>0)))
            self.k['gather_block'](((cap*cap+255)//256,),(256,),(self.W,self.b,self.ids,self.meta,self.C,self.rhs,self.weights,np.int32(cap)))
            self.correct((1,),(128,),(self.C,self.rhs,self.weights,self.weights,self.inner,self.meta,self.L,
                         np.int32(cap),np.float64(tolerance),np.int32(1000),np.int32(self.local_sweeps)),
                         shared_mem=(6*cap+128)*8+2*cap*4)
            self.k['check_all']((nb,),(256,),(self.W,self.b,self.weights,self.ids,self.meta,self.partial,self.worst,np.int32(cap)))
            self.k['next_column']((1,),(32,),(self.partial,self.worst,self.inner,self.meta,self.ids,self.weights,np.int32(nb),np.int32(cap),np.float64(tolerance)))
        self.meta[4]=1;expansion();cp.cuda.get_current_stream().synchronize()
        stream=cp.cuda.Stream(non_blocking=True)
        with stream:
            stream.begin_capture()
            for _ in range(self.batch):expansion()
            self.graph=stream.end_capture()
        self.rebuilds+=1

    def solve(self,b,seeds=None,tolerance=1e-14):
        self.calls+=1;m=len(b)
        if not m:return cp.zeros(0),dict(outer_iterations=0,stored_columns=0)
        if not isinstance(seeds,tuple):
            seeds=seeds or {};seeds=(cp.asarray(list(seeds),dtype=cp.int64),cp.asarray(list(seeds.values())))
        if len(seeds[0])>=self.cap:return self.fallback(b,seeds,tolerance,5)
        nnz=max(len(B.data) for B,C,v in self.device.blocks)
        signature=tuple((int(C.data.ptr),C.shape,C.strides) for B,C,v in self.device.blocks)
        if self.graph is None or m>self.mcap or nnz>self.nnzcap or signature!=self.operator_signature or tolerance!=self.tolerance:
            self.allocate(m,nnz,tolerance)
        self.b[:m]=b;self.weights.fill(0);self.ids[:len(seeds[0])]=seeds[0];self.weights[:len(seeds[1])]=seeds[1]
        self.meta.set(np.asarray([m,len(seeds[0]),0,0,0,0,0],dtype=np.int32))
        for (B,C,v),(ptr,col,val,rank,_,T) in zip(self.device.blocks,self.buffers):
            ptr[:m+1]=B.indptr;col[:len(B.data)]=B.indices;val[:len(B.data)]=B.data;rank[:m]=v
        self.k['set_start']((1,),(32,),(self.b,self.meta,self.ids,self.weights,np.int32(self.cap)))
        for _ in range((m+self.batch-1)//self.batch):
            self.graph.launch();state=cp.asnumpy(self.meta);self.host_reads+=1
            if state[5]:
                self.expansions+=int(state[6])
                return self.fallback(b,seeds,tolerance,int(state[5]))
            if state[4]:
                lam=cp.zeros(m)
                self.k['scatter_result']((1,),(self.cap,),(self.ids,self.weights,self.meta,lam,np.int32(self.cap)))
                self.device.statistics['device_response_columns']+=int(state[2]);self.expansions+=int(state[6])
                self.local_predictor_sweeps+=int(state[6])*self.local_sweeps
                return lam,dict(outer_iterations=int(state[6]),stored_columns=int(state[2]),parallel=True)
        raise RuntimeError('Parallel working set unresolved')

    def fallback(self,b,seeds,tolerance,reason):
        self.fallbacks+=1;self.fallback_reasons[str(reason)]=self.fallback_reasons.get(str(reason),0)+1
        # Rare rank-deficient/capacity path. cuSOLVER equality solve plus null-space
        # handling; no change to tolerances or diagonal contact regularization.
        ids=cp.asnumpy(seeds[0]).tolist();weights=seeds[1].copy();j=int(cp.argmin(b))
        if j not in ids:ids.append(j);weights=cp.concatenate((weights,cp.zeros(1)))
        response=None
        for outer in range(len(b)):
            if len(ids)>256:raise MemoryError('Parallel contact fallback capacity')
            new=ids if response is None else ids[-1:]
            col=self.device.columns(cp.asarray(new,dtype=cp.int64))
            response=col if response is None else cp.concatenate((response,col),axis=1)
            dids=cp.asarray(ids,dtype=cp.int64)
            weights,_=lcp(response[dids],b[dids],tolerance,weights)
            gap=b+response@weights;ae=float(cp.min(gap[dids]));eligible=gap.copy();eligible[dids]=cp.inf
            j=int(cp.argmin(eligible))
            if float(eligible[j])>=-10*tolerance:
                if ae<-100*tolerance:raise RuntimeError('Parallel fallback certificate failed')
                lam=cp.zeros(len(b));lam[dids]=weights
                return lam,dict(outer_iterations=outer+1,stored_columns=len(ids),fallback=reason)
            ids.append(j);weights=cp.concatenate((weights,cp.zeros(1)))
        raise RuntimeError('Parallel contact fallback unresolved')
