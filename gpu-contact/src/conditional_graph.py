"""CUDA C++ graph ownership and conditional WHILE bridge; no numerical changes."""
import ctypes as ct
import hashlib
import os
from pathlib import Path
import subprocess
import cupy as cp
from validate_mechanisms import ROOT,guard

_library=None

class _CaptureWorkspaceRequest(RuntimeError):
    """Request backing memory outside graph capture, without calling cudaMalloc."""
    def __init__(self,size):
        self.size=int(size)
        super().__init__(f'Capture workspace needs {self.size} bytes')

def library():
    from portable_graph import library as load
    return load()

def checked(code):
    if code:raise RuntimeError(f'CUDA graph {code}: '+library().dc_error(code).decode())

class NativeGraph:
    def __init__(self):self.graph=ct.c_void_p();self.exec=ct.c_void_p();self.refs=[]
    @classmethod
    def capture(cls,fn,pooled=False):
        obj=cls();lib=library();stream=cp.cuda.Stream(non_blocking=True)
        if pooled:
            passes=int(os.environ.get('CONTACT_CAPTURE_WARMUP_PASSES','5'))
            if not 2<=passes<=5:raise ValueError('Capture warm-up passes must be between 2 and 5')
            state={'capturing':False};reserve={};requests=[]
            def allocate(size):
                if state['capturing']:raise _CaptureWorkspaceRequest(size)
                return cp.cuda.MemoryPointer(cp.cuda.Memory(size),0)
            pool=cp.cuda.MemoryPool(allocate);obj.refs.append(pool)
            with cp.cuda.using_allocator(pool.malloc):
                with stream:
                    for attempt in range(16):
                        for _ in range(passes):fn()
                        # Retained outputs can change the pool's free size bins
                        # between warm-up and capture. Back those requests here,
                        # before cudaStreamBeginCapture, never inside capture.
                        buffers=[pool.malloc(size) for size,count in reserve.items() for _ in range(count)]
                        del buffers
                        stream.synchronize()
                        checked(lib.dc_begin(stream.ptr));state['capturing']=True
                        try:fn()
                        except _CaptureWorkspaceRequest as demand:
                            state['capturing']=False
                            code=lib.dc_end(stream.ptr,ct.byref(obj.graph))
                            obj.close();checked(code)
                            requests.append(demand.size)
                            reserve[demand.size]=reserve.get(demand.size,0)+4
                            continue
                        except Exception:
                            state['capturing']=False
                            lib.dc_end(stream.ptr,ct.byref(obj.graph));raise
                        finally:state['capturing']=False
                        checked(lib.dc_end(stream.ptr,ct.byref(obj.graph)))
                        obj.capture_workspace=dict(retries=attempt,requests=requests,
                            reserve_bytes=sum(s*n for s,n in reserve.items()),warmup_passes=passes,
                            warmup_executions=passes*(attempt+1))
                        break
                    else:raise RuntimeError('Capture workspace planning exceeded 16 attempts')
            return obj
        with stream:
            checked(lib.dc_begin(stream.ptr))
            try:fn()
            except Exception:
                lib.dc_end(stream.ptr,ct.byref(obj.graph));raise
            checked(lib.dc_end(stream.ptr,ct.byref(obj.graph)))
        return obj
    def instantiate(self):checked(library().dc_instantiate(self.graph,ct.byref(self.exec)));return self
    def while_loop(self,state,done=4,error=5,count=6,limit=4096):
        obj=NativeGraph();obj.refs=[self,state]
        checked(library().dc_while(self.graph,state.data.ptr,done,error,count,limit,ct.byref(obj.graph),ct.byref(obj.exec)))
        return obj
    def launch(self):checked(library().dc_launch(self.exec,cp.cuda.get_current_stream().ptr))
    @classmethod
    def nested(cls,parts,outer,inner,outer_limit=5000,inner_limit=4096):
        obj=cls();obj.refs=[*parts,outer,inner]
        values=(ct.c_void_p*5)(*(p.graph.value for p in parts))
        checked(library().dc_nested(values,outer.data.ptr,inner.data.ptr,outer_limit,inner_limit,ct.byref(obj.graph),ct.byref(obj.exec)))
        return obj
    def close(self):
        if self.graph.value or self.exec.value:
            checked(library().dc_destroy(self.graph,self.exec));self.graph=ct.c_void_p();self.exec=ct.c_void_p()
    def __del__(self):
        try:self.close()
        except Exception:pass
