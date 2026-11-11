"""Migration adapter for native exact refill; not yet a standalone solver.

Python retains model setup and regional policy. One native call performs a whole
refill, including packed gather, bounded transfers, BLAS correction and scatter.
All geometry/HHT equations remain in the accepted CUDA graph. No new FE solves.
"""
import ctypes as ct
import sys
from pathlib import Path
import numpy as np
import cupy as cp

ROOT=Path(__file__).resolve().parents[1]

class Stats(ct.Structure):
    _fields_=[(k,ct.c_uint64) for k in ('calls','chunks','payload_bytes','factor_bytes','host_bytes','device_bytes')]+[
        (k,ct.c_double) for k in ('gather_s','wait_s','total_s')]

def library():
    name='coflex_native.dll' if sys.platform=='win32' else 'libcoflex_native.so'
    lib=ct.CDLL(str(ROOT/'cache/native_cpp'/name))
    p=ct.c_void_p;i=ct.c_int;u=ct.c_uint64
    signatures={
        'cf_last_error':([],ct.c_char_p), 'cf_create':([ct.c_char_p,i,i,u,ct.POINTER(p)],i),
        'cf_destroy':([p],i), 'cf_register_bank':([p,p,u,p,p,i,i,p,i,ct.POINTER(i)],i),
        'cf_gather':([p,i,p,i,p,i,p],i),
        'cf_refill':([p,i,u,u,u,i,p,i,p,i,p,p],i), 'cf_stats':([p,ct.POINTER(Stats),i],i),
        'cf_graph_batch':([u,u,u,u,ct.c_double,i,u,i,p,p],i)}
    for name,(args,result) in signatures.items():
        f=getattr(lib,name);f.argtypes=args;f.restype=result
    return lib

class NativeRefill:
    def __init__(self,slot_bytes=4*2**20,max_rows=8192,max_rank=900):
        self.lib=library();self.handle=ct.c_void_p();self.sources={};self.references=[]
        self.max_rows=int(max_rows);self.max_rank=int(max_rank);self.slot_bytes=slot_bytes
        self.check(self.lib.cf_create(str(ROOT/'cache/native_cpp/images').encode(),self.max_rows,self.max_rank,
            slot_bytes,ct.byref(self.handle)))
        self.statistics={}  # Existing diagnostic harness may reset this dict.
    def check(self,code):
        if code:raise RuntimeError(self.lib.cf_last_error().decode())
    def register(self,source,query):
        key=id(source)
        if key not in self.sources:
            U=getattr(source,'U',None);rank=0 if U is None else U.shape[1]
            if U is not None and (U.dtype!=np.float64 or not U.flags.c_contiguous):
                raise ValueError('Native source correction must be contiguous FP64')
            bank=ct.c_int()
            self.check(self.lib.cf_register_bank(self.handle,query.data.ctypes.data,query.data.size,
                query.offsets.ctypes.data,query.widths.ctypes.data,query.n,query.t,
                None if U is None else U.ctypes.data,rank,ct.byref(bank)))
            self.sources[key]=bank.value;self.references.append((source,query,U))
        return self.sources[key]
    def update(self,B,source,query,all_ids,missing,occupied,destinations):
        if not B.C.flags.f_contiguous or not B.regional_schur_factor.flags.c_contiguous:
            raise ValueError('Native regional layout mismatch')
        arrays=[np.ascontiguousarray(x,dtype=np.int64) for x in (all_ids,missing,occupied,destinations)]
        a,b,c,d=arrays
        self.check(self.lib.cf_refill(self.handle,self.register(source,query),cp.cuda.get_current_stream().ptr,
            B.C.data.ptr,B.regional_schur_factor.data.ptr,len(B.C),a.ctypes.data,len(a),b.ctypes.data,len(b),c.ctypes.data,d.ctypes.data))
    def ledger(self,reset=False):
        s=Stats();self.check(self.lib.cf_stats(self.handle,ct.byref(s),int(reset)))
        values={k:getattr(s,k) for k,_ in s._fields_}
        return dict(native=True,pinned_host_bytes=s.host_bytes,device_staging_and_workspace_bytes=s.device_bytes,
            statistics=values,scope='Native-owned staging/workspace only; CuPy pools, regional matrices and mapped file pages additional')
    def close(self):
        if self.handle.value:self.check(self.lib.cf_destroy(self.handle));self.handle=ct.c_void_p()
    def __del__(self):
        try:self.close()
        except Exception:pass

def install_refill(sim):
    """Install before restore/rebuild, after geometry construction. No defaults change."""
    if hasattr(sim,'refill_staging'):
        cp.cuda.get_current_stream().synchronize();del sim.refill_staging
    rank=max((getattr(s,'U',np.empty((0,0))).shape[1] for s in sim.sources),default=0)
    # Every regional square fits in the existing declared budget. This bounds
    # scratch independently of one scene, without pre-reading future contacts.
    capacity=int(np.sqrt(sim.window_budget/8))+1
    sim.refill_staging=NativeRefill(sim.scene.get('regional_refill_slot_bytes',4*2**20),capacity,rank)
    return sim.refill_staging

class NativeBatch:
    """C++ graph submission/status path; regional retry policy is still Python."""
    def __init__(self,sim):
        self.lib=library();self.state=np.zeros(3,dtype=np.int32)
        self.rows=np.empty(sim.device_history.shape,dtype=np.float64)
    def launch(self,sim,steps):
        if self.rows.shape!=sim.device_history.shape:
            self.rows=np.empty(sim.device_history.shape,dtype=np.float64)
        code=self.lib.cf_graph_batch(sim.trajectory.exec.value,cp.cuda.get_current_stream().ptr,
            sim.run_state.data.ptr,sim.clock.data.ptr,sim.t+sim.dt,steps,sim.device_history.data.ptr,
            self.rows.shape[1],self.rows.ctypes.data,self.state.ctypes.data)
        if code:raise RuntimeError(self.lib.cf_last_error().decode())
        done,error=int(self.state[1]),int(self.state[2])
        return done,error,self.rows[:done].copy()
