"""Exact compiled gather from the existing symmetric tile bank (no new bank).

The packed data remains a read-only mmap. A small offset table removes Python
tile loops and their repeated masks/temporary advanced-index arrays.
"""
import time
import numpy as np
from numba import njit

@njit(nogil=True,cache=False)
def gather(data,offsets,widths,rows,cols,t,out):
    for i in range(len(rows)):
        r=rows[i];a=r//t;ri=r%t
        for j in range(len(cols)):
            c=cols[j];b=c//t;ci=c%t
            if a<=b:idx=offsets[a,b]+ri*widths[b]+ci
            else:idx=offsets[b,a]+ci*widths[a]+ri
            out[i,j]=data[idx]

class PackedSurfaceQuery:
    def __init__(self,source):
        self.source=source;self.n=source.n;self.t=source.tile_size
        self.data=source.backing
        if self.data is None:raise ValueError('Compiled gather requires the canonical packed mmap')
        nt=(self.n+self.t-1)//self.t
        self.offsets=np.zeros((nt,nt),dtype=np.int64)
        self.widths=np.minimum(self.t,self.n-np.arange(nt)*self.t).astype(np.int64)
        origin=self.data.ctypes.data
        for (a,b),tile in source.tiles.items():
            if not tile.flags.c_contiguous or tile.dtype!=np.float64:raise ValueError('Unexpected packed tile')
            self.offsets[a,b]=(tile.ctypes.data-origin)//8
        # Compilation belongs to initialization, never to an online miss.
        self.block(np.array([0],dtype=np.int64),np.array([0],dtype=np.int64))
        self.calls=0;self.query_s=0.;self.returned_bytes=0

    def block(self,rows,cols):
        rows=np.asarray(rows,dtype=np.int64).ravel();cols=np.asarray(cols,dtype=np.int64).ravel()
        out=np.empty((len(rows),len(cols)),dtype=np.float64)
        return self.block_into(rows,cols,out)

    def block_into(self,rows,cols,out):
        """Exact gather into caller-owned contiguous storage (including pinned RAM)."""
        tick=time.perf_counter();rows=np.asarray(rows,dtype=np.int64).ravel();cols=np.asarray(cols,dtype=np.int64).ravel()
        if np.any(rows<0) or np.any(rows>=self.n) or np.any(cols<0) or np.any(cols>=self.n):raise IndexError('Outside designated surface')
        if out.shape!=(len(rows),len(cols)) or out.dtype!=np.float64 or not out.flags.c_contiguous or not out.flags.writeable:
            raise ValueError('Expected writable contiguous FP64 output of the requested shape')
        gather(self.data,self.offsets,self.widths,rows,cols,self.t,out)
        if hasattr(self,'calls'):
            self.calls+=1;self.query_s+=time.perf_counter()-tick;self.returned_bytes+=out.nbytes
        return out

def prepare(source):
    base=source.source if hasattr(source,'U') else source
    if not hasattr(base,'packed_query'):base.packed_query=PackedSurfaceQuery(base)
    return base.packed_query
