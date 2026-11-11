"""Dimension-dispatched parallel rows; frozen small-coordinate path retained."""
from pathlib import Path
import numpy as np
import cupy as cp
from clock_modal import ClockModalKernels

class ParallelModalKernels(ClockModalKernels):
    def __init__(self):
        super().__init__()
        self.serial_modal=self.modal
        self.parallel_module=cp.RawModule(code=Path(__file__).with_suffix('.cu').read_text())
        self.stages=[self.parallel_module.get_function('modal_'+s) for s in ('rows','reference','inverse','finish')]
        self.parallel_map=self.parallel_module.get_function('parallel_map')
        self.parallel_direction=self.parallel_module.get_function('parallel_direction')
        self.modal=self.launch_modal

    def launch_modal(self,grid,block,args,shared_mem=0):
        n=int(args[22])
        if n<64:return self.serial_modal(grid,block,args,shared_mem=shared_mem)
        da=cp.empty(n);y=cp.empty(n);extended=args+(da,y)
        self.stages[0](((n+3)//4,),(128,),extended)
        self.stages[1]((1,),(32,),extended)
        self.stages[2](((n+3)//4,),(128,),extended)
        self.stages[3]((1,),(256,),extended)

    def map_body(self,B,q,s,x):
        if B.P.shape[1]<32:return super().map_body(B,q,s,x)
        b=B.source;n=len(B.X);R=cp.empty((3,3));tangent=cp.empty(B.X.size)
        self.parallel_map(((n+3)//4,),(128,),(B.X,B.P,q,s,B.axis,B.center,R,x,tangent,np.int32(n),np.int32(B.P.shape[1]),
            np.int32(b.offset),np.int32(b.revolute),np.int32(b.prismatic)))
        return R,tangent if b.reference else None

    def apply_body(self,B,dd,ds,R,tangent,out):
        if B.P.shape[1]<32:return super().apply_body(B,dd,ds,R,tangent,out)
        b=B.source;n=len(B.X)
        self.parallel_direction(((n+3)//4,),(128,),(B.P,dd,ds,R,tangent if tangent is not None else cp.empty(0),out,
            np.int32(n),np.int32(B.P.shape[1]),np.int32(b.offset)))
