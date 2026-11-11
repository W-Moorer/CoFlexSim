"""Device finite triangular overlap, P1 integration and total derivatives."""
from pathlib import Path
import numpy as np
import cupy as cp


class DeviceOverlap:
    def __init__(self):
        self.module=cp.RawModule(code=Path(__file__).with_name('overlap.cu').read_text(),options=('--std=c++11',))
        self.values_kernel=self.module.get_function('overlap_values')
        self.derivative_kernel=self.module.get_function('overlap_derivatives')

    def build(self,S,M,radius,band,asymmetric=False,derivatives=True):
        S=cp.ascontiguousarray(cp.asarray(S,dtype=cp.float64));M=cp.ascontiguousarray(cp.asarray(M,dtype=cp.float64));n=len(S)
        area=cp.zeros((n,2,3));gap=cp.zeros_like(area);points=cp.zeros((n,2,3,3));info=cp.zeros((n,2,6))
        samples=cp.zeros((n,2),dtype=cp.int32);counts=cp.zeros(n,dtype=cp.int32)
        vg=cp.zeros((n,12));vp=cp.zeros((n,12,3))
        if n:
            self.values_kernel(((n+63)//64,),(64,),(S,M,np.int32(n),np.float64(radius),np.float64(band),np.int32(asymmetric),
                area,gap,points,info,samples,counts,vg,vp))
        selected=cp.flatnonzero(samples[:,0]).astype(cp.int32)
        partial=vertex=None
        if derivatives:
            partial=cp.zeros((len(selected),6,18));vertex=cp.zeros((len(selected),12,18))
            if len(selected):
                self.derivative_kernel((len(selected),),(32,),(S,M,selected,np.int32(len(selected)),np.float64(radius),
                        np.float64(band),np.int32(asymmetric),samples,partial,vertex))
        return dict(selected=selected,area=area[selected],gap=gap[selected],points=points[selected],info=info[selected],
                    samples=samples[selected],counts=counts[selected],vg=vg[selected],vp=vp[selected],partial=partial,vertex=vertex)
