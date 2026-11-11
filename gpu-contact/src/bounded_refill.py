"""Exact regional refill through two bounded pinned-RAM/device staging slots.

CPU gathers the next chunk while the GPU consumes the preceding chunk. Events
protect both host and device slot reuse. The current stream owns all operations;
the caller synchronizes before accepting a new response window. No new library,
global FE solve, approximate coefficient, or dropped cross-region term.
"""
import time
import numpy as np
import cupy as cp


class BoundedRefill:
    def __init__(self,slot_bytes=4*2**20):
        self.slot_bytes=int(slot_bytes)//8*8
        if self.slot_bytes<8:raise ValueError('Refill staging budget is too small')
        self.pinned=[cp.cuda.alloc_pinned_memory(self.slot_bytes) for _ in range(2)]
        self.host=[np.frombuffer(p,dtype=np.float64,count=self.slot_bytes//8) for p in self.pinned]
        self.device=[cp.empty(self.slot_bytes//8,dtype=cp.float64) for _ in range(2)]
        self.events=[cp.cuda.Event(disable_timing=True) for _ in range(2)]
        self.used=[False,False]
        self.statistics=dict(chunks=0,payload_bytes=0,gather_s=0.,slot_wait_s=0.,
                             enqueue_wall_s=0.,max_request_bytes=0)

    def update(self,B,source,query,all_ids,missing,occupied,destinations):
        if not hasattr(query,'block_into'):raise TypeError('Bounded refill requires an exact packed block_into query')
        rows=len(all_ids);cols=len(missing)
        if not cols:return
        width=self.slot_bytes//(8*max(1,rows))
        if width<1:raise MemoryError('One regional column exceeds the declared refill slot budget')
        st=self.statistics;st['max_request_bytes']=max(st['max_request_bytes'],8*rows*cols)
        ii=cp.asarray(occupied);jj=cp.asarray(destinations);kk=cp.asarray(missing)
        rank=B.regional_schur_factor.shape[1]
        if rank:
            if hasattr(source,'device_factor'):B.regional_schur_factor[jj]=source.device_factor[kk]
            else:B.regional_schur_factor[jj]=cp.asarray(source.U[missing])
            left=B.regional_schur_factor[ii]
        stream=cp.cuda.get_current_stream()
        for start in range(0,cols,width):
            end=min(cols,start+width);slot=st['chunks']%2;size=rows*(end-start)
            tick=time.perf_counter()
            if self.used[slot]:self.events[slot].synchronize()
            st['slot_wait_s']+=time.perf_counter()-tick
            h=self.host[slot][:size].reshape(rows,end-start)
            tick=time.perf_counter();query.block_into(all_ids,missing[start:end],h)
            st['gather_s']+=time.perf_counter()-tick
            tick=time.perf_counter();v=self.device[slot][:size].reshape(rows,end-start)
            v.set(h,stream=stream)
            j=jj[start:end]
            if rank:v+=left@B.regional_schur_factor[j].T
            B.C[cp.ix_(ii,j)]=v;B.C[cp.ix_(j,ii)]=v.T
            self.events[slot].record(stream);self.used[slot]=True
            st['chunks']+=1;st['payload_bytes']+=size*8
            st['enqueue_wall_s']+=time.perf_counter()-tick

    def ledger(self):
        return dict(pinned_host_bytes=2*self.slot_bytes,device_staging_bytes=2*self.slot_bytes,
                    statistics=self.statistics.copy(),
                    scope='Staging only; regional matrices, factors, GEMM temporaries and OS file pages are additional')
