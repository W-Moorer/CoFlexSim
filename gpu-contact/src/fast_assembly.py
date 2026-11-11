"""Fused CUDA support integration and direct CSR emission (duplicates allowed).

No global COO sorting/coalescing is required: contact actions sum duplicate
element contributions exactly as ordinary FE assembly does. Explicit CSR rows
retain complete xyz triplets for the material-frame transformation.
"""
from pathlib import Path
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as csp


class FastAssembly:
    def __init__(self):
        self.module=cp.RawModule(code=Path(__file__).with_name('geometry_assembly.cu').read_text())
        self.k={n:self.module.get_function(n) for n in ['support_sum','support_rows','count_entries','assemble_rows']}

    @staticmethod
    def run(kernel,n,args):
        if n:kernel(((n+127)//128,),(128,),args)

    def build(self,data,owners,mids,nodes,x,nf,size,maxnodes,paircode,maxface,guardbase,offset,radius,derivatives):
        n=len(owners);nodes=cp.ascontiguousarray(nodes,dtype=cp.int32);owners=owners.astype(cp.int32);mids=mids.astype(cp.int32)
        coverage=cp.zeros((2,nf));area=cp.zeros(maxnodes);numer=cp.zeros(maxnodes)
        self.run(self.k['support_sum'],n,(owners,nodes,data['samples'],data['area'],data['gap'],np.int32(n),np.int32(nf),coverage,area,numer))
        keep=cp.flatnonzero(area>1e-24);nw=len(keep)
        voff=cp.concatenate((cp.zeros(1,dtype=cp.int32),cp.cumsum(data['counts'],dtype=cp.int32)))
        ng=int(voff[-1]);m=nw+ng;lookup=cp.full(maxnodes,-1,dtype=cp.int32)
        gaps=cp.empty(m);keys=cp.empty(m,dtype=cp.int64)
        self.run(self.k['support_rows'],nw,(keep,np.int32(nw),area,numer,lookup,gaps,keys,np.int64(paircode*maxnodes)))
        if derivatives:
            counts=cp.zeros(m,dtype=cp.int32)
            self.run(self.k['count_entries'],max(n,ng),(nodes,data['samples'],lookup,np.int32(n),np.int32(nw),np.int32(ng),counts))
            ptr=cp.concatenate((cp.zeros(1,dtype=cp.int32),cp.cumsum(counts,dtype=cp.int32)))
            nnz=int(ptr[-1]);cols=cp.empty(nnz,dtype=cp.int32);vals=cp.empty(nnz);cursor=ptr[:-1].copy()
            partial=data['partial'];vertex=data['vertex']
        else:
            ptr=cursor=cols=cp.empty(0,dtype=cp.int32);vals=partial=vertex=cp.empty(0)
        error=cp.zeros(1,dtype=cp.int32)
        self.run(self.k['assemble_rows'],n,(owners,mids,nodes,data['samples'],data['area'],data['info'],coverage,area,numer,lookup,
                partial,vertex,data['vg'],data['vp'],data['counts'],voff,x,np.int32(n),np.int32(nf),np.int32(nw),
                np.int32(maxface),np.int32(paircode),np.int64(guardbase),np.float64(offset),np.float64(radius),
                cursor,ptr,cols,vals,gaps,keys,error,np.int32(derivatives)))
        code=int(error[0])
        if code:raise RuntimeError('GPU contact coverage error' if code==1 else 'GPU contact negative search boundary')
        G=csp.csr_matrix((vals,cols,ptr),shape=(m,size)) if derivatives else None
        physical=cp.min(gaps[nw:])-offset if ng else cp.asarray(0.)
        return gaps,G,keys,physical,nw
