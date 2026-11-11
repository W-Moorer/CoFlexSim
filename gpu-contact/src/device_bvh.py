"""Reference-built AABB hierarchy; all online refit/query operations on CUDA."""
from pathlib import Path
import numpy as np
import cupy as cp


class DeviceBVH:
    module=None
    def __init__(self,X,faces,leaf_size=8):
        if DeviceBVH.module is None:
            DeviceBVH.module=cp.RawModule(code=Path(__file__).with_name('bvh.cu').read_text())
        self.kernels={n:self.module.get_function(n) for n in ['triangle_boxes','leaf_boxes','inner_boxes','query_boxes']}
        faces=np.asarray(faces,dtype=np.int32);centers=np.asarray(X)[faces].mean(1)
        nodes=[];order=[];levels={};leaves=[]
        def build(ids,depth):
            i=len(nodes);nodes.append([-1,-1,0,0])
            if len(ids)<=leaf_size:
                nodes[i][2:]=[len(order),len(ids)];order.extend(ids);leaves.append(i)
            else:
                axis=np.ptp(centers[ids],axis=0).argmax();ids=ids[np.argsort(centers[ids,axis],kind='stable')];mid=len(ids)//2
                left=build(ids[:mid],depth+1);right=build(ids[mid:],depth+1);nodes[i][:2]=[left,right];levels.setdefault(depth,[]).append(i)
            return i
        build(np.arange(len(faces),dtype=np.int32),0)
        self.faces=cp.asarray(faces);self.nodes=cp.asarray(nodes,dtype=cp.int32);self.order=cp.asarray(order,dtype=cp.int32)
        self.leaves=cp.asarray(leaves,dtype=cp.int32);self.levels=[cp.asarray(levels[k],dtype=cp.int32) for k in sorted(levels,reverse=True)]
        self.lo=cp.empty((len(nodes),3));self.hi=cp.empty_like(self.lo);self.tlo=cp.empty((len(faces),3));self.thi=cp.empty_like(self.tlo)
        self.refits=0;self.queries=0;self.refit_graph=None;self.query_cache={}

    @staticmethod
    def launch(kernel,n,args):
        if n:kernel(((n+127)//128,),(128,),args)

    def refit(self,x):
        if self.refit_graph is None:
            self.refit_input=cp.empty_like(x)
            def dispatch():
                n=len(self.faces);self.launch(self.kernels['triangle_boxes'],n,(self.refit_input,self.faces,self.tlo,self.thi,np.int32(n)))
                self.launch(self.kernels['leaf_boxes'],len(self.leaves),(self.nodes,self.order,self.leaves,np.int32(len(self.leaves)),self.tlo,self.thi,self.lo,self.hi))
                for ids in self.levels:self.launch(self.kernels['inner_boxes'],len(ids),(self.nodes,ids,np.int32(len(ids)),self.lo,self.hi))
            cp.copyto(self.refit_input,x);dispatch();cp.cuda.get_current_stream().synchronize()
            stream=cp.cuda.Stream(non_blocking=True)
            with stream:
                stream.begin_capture();dispatch();self.refit_graph=stream.end_capture()
        cp.copyto(self.refit_input,x);self.refit_graph.launch()
        self.refits+=1

    def query(self,x,faces,radius):
        faces=cp.ascontiguousarray(faces,dtype=cp.int32);n=len(faces)
        cap=128
        while True:
            key=(int(faces.data.ptr),n,cap,float(radius),x.shape)
            if key not in self.query_cache:
                inp=cp.empty_like(x);lo=cp.empty((n,3));hi=cp.empty_like(lo)
                out=cp.empty((n,cap),dtype=cp.int32);counts=cp.empty(n,dtype=cp.int32);overflow=cp.empty(1,dtype=cp.int32)
                def dispatch():
                    out.fill(-1);counts.fill(0);overflow.fill(0)
                    self.launch(self.kernels['triangle_boxes'],n,(inp,faces,lo,hi,np.int32(n)))
                    self.launch(self.kernels['query_boxes'],n,(self.nodes,self.order,self.lo,self.hi,self.tlo,self.thi,lo,hi,np.int32(n),np.float64(radius),np.int32(cap),out,counts,overflow))
                cp.copyto(inp,x);dispatch();cp.cuda.get_current_stream().synchronize()
                stream=cp.cuda.Stream(non_blocking=True)
                with stream:
                    stream.begin_capture();dispatch();graph=stream.end_capture()
                self.query_cache[key]=(inp,lo,hi,out,counts,overflow,graph,faces)
            inp,lo,hi,out,counts,overflow,graph,_=self.query_cache[key]
            cp.copyto(inp,x);graph.launch()
            code=int(overflow[0])
            if code==2:raise RuntimeError('BVH traversal stack overflow')
            if not code:break
            cap*=2
            if cap>len(self.faces)*2:raise RuntimeError('BVH candidate buffer overflow')
        owners,slots=cp.nonzero(out>=0);mids=out[owners,slots]
        order=cp.lexsort(cp.stack((mids,owners)));self.queries+=1
        return owners[order].astype(cp.int32),mids[order]
