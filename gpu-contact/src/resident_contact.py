"""Reusable surface-contact maps and response graphs; unchanged FP64 algebra.

The workspace depends on body operators and capacity, never on scene names.
Moving geometry is copied every evaluation; no stale contact matrix is reused.
"""
from types import SimpleNamespace
import numpy as np
import cupy as cp
from working_graph import bucket

CODE=r'''
extern "C" __global__ void matvec(const double*A,const double*x,double*y,int rows,int cols,int trans){
 int r=blockIdx.x,t=threadIdx.x,n=trans?rows:cols;double sum=0.;
 for(int j=t;j<n;j+=blockDim.x)sum+=A[trans?(long long)j*cols+r:(long long)r*cols+j]*x[j];
 __shared__ double work[128];work[t]=sum;__syncthreads();
 for(int s=64;s;s/=2){if(t<s)work[t]+=work[t+s];__syncthreads();}if(!t)y[r]=work[0];
}
extern "C" __global__ void force_ids(const double*f,int*ids,int*count,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&f[i]!=0.)ids[atomicAdd(count,1)]=i;
}
extern "C" __global__ void sparse_response(const double*R,const double*f,const int*ids,const int*count,double*out,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;double sum=0.;
 for(int j=0;j<count[0];j++){int c=ids[j];sum+=R[(long long)c*n+i]*f[c];}out[i]=sum;
}
'''

class ResidentContact:
    def __init__(self,sim):
        self.sim=sim;self.mcap=0;self.ecap=0;self.rebuilds=0
        mod=cp.RawModule(code=CODE);self.pack=mod.get_function('force_ids');self.response=mod.get_function('sparse_response');self.mv_kernel=mod.get_function('matvec')
        self.refs=[]

    def mv(self,A,x,transpose=False):
        rows,cols=A.shape;y=cp.empty(cols if transpose else rows)
        assert x.size==(rows if transpose else cols),(A.shape,x.shape,transpose)
        assert A.dtype==cp.float64 and x.dtype==cp.float64
        if not len(y):return y
        self.mv_kernel((len(y),),(128,),(A,x,y,np.int32(rows),np.int32(cols),np.int32(transpose)))
        return y

    def capture(self,fn):
        fn();cp.cuda.get_current_stream().synchronize()
        stream=cp.cuda.Stream(non_blocking=True)
        with stream:
            stream.begin_capture();fn();graph=stream.end_capture()
        graph.launch();cp.cuda.get_current_stream().synchronize()
        return graph

    def allocate(self,m,e):
        sim=self.sim;self.mcap=max(self.mcap,bucket(m));self.ecap=max(self.ecap,3*bucket((e+2)//3))
        mc,ec=self.mcap,self.ecap
        self.ptr=cp.zeros(mc+1,dtype=cp.int32);self.cols=cp.zeros(ec,dtype=cp.int32);self.values=cp.zeros(ec)
        self.G=SimpleNamespace(indptr=self.ptr,indices=self.cols,data=self.values,shape=(mc,sim.ns))
        self.lam=cp.zeros(mc);self.free=cp.zeros(sim.nd+sim.ns);self.gaps=cp.zeros(mc)
        self.rot=[cp.eye(3) for b in sim.bodies]
        self.tangent=[cp.zeros(b.X.size) if b.reference else None for b in sim.bodies]
        def prepare():
            self.blocks=[];self.contacts=[];self.refs=[];mapped=cp.empty(sim.ns)
            for i,(b,B,R,tangent) in enumerate(zip(sim.bodies,sim.dynamic,self.rot,self.tangent)):
                sl,sh=sim.soff[i:i+2];dl,dh=sim.doff[i:i+2]
                pu=self.mv(B.P,B.u) if b.reference else cp.empty(0);self.refs.append(pu)
                rows,jr,v,rank=sim.actions.body(self.G,R,tangent,pu,B.sigma if b.reference else cp.asarray(1.),int(sl),int(sh-sl))
                self.blocks.append((rows,B.C,rank));self.contacts.append((rows,jr,v))
                sim.modal_kernels.apply_body(B,self.free[dl:dh],self.free[sim.nd+sl:sim.nd+sh],R,tangent,mapped[sl:sh])
            self.free_gap=self.gaps+sim.actions.action(self.G,mapped);self.refs.append(mapped)
        self.prepare_graph=self.capture(prepare)
        def recover():
            self.direction=self.free.copy();self.targets=cp.zeros(sim.ns);self.forces=cp.zeros(sim.ns);self.targetd=cp.zeros(sim.nd);self.recover_refs=[]
            for i,(b,B,(rows,jr,v)) in enumerate(zip(sim.bodies,sim.dynamic,self.contacts)):
                sl,sh=sim.soff[i:i+2];dl,dh=sim.doff[i:i+2]
                f=sim.actions.adjoint(rows,self.lam);fm=self.mv(B.P,f,True);y=self.mv(B.inverse,fm)
                self.forces[sl:sh]=f;self.targetd[dl+b.offset:dh]=fm
                if b.reference:
                    theta=cp.sum(v*self.lam)/B.sigma;self.direction[dl]+=theta;y=y-B.u*theta;self.targetd[dl]=cp.sum(jr*self.lam)
                    self.recover_refs.append(theta)
                self.direction[dl+b.offset:dh]+=y
                if B.R is not None:
                    n=int(sh-sl);ids=cp.empty(n,dtype=cp.int32);count=cp.zeros(1,dtype=cp.int32)
                    self.pack(((n+127)//128,),(128,),(f,ids,count,np.int32(n)))
                    self.response(((n+127)//128,),(128,),(B.R,f,ids,count,self.targets[sl:sh],np.int32(n)))
                    self.recover_refs.extend((ids,count))
                self.recover_refs.extend((f,fm,y))
        self.recover_graph=self.capture(recover);self.rebuilds+=1

    def prepare(self,world,mapping,free):
        m=len(world.gaps);e=len(world.G.data)
        if m>self.mcap or e>self.ecap or not self.mcap:self.allocate(max(m,1),e)
        self.ptr[:m+1]=world.G.indptr;self.ptr[m+1:]=world.G.indptr[-1]
        self.cols[:e]=world.G.indices;self.values[:e]=world.G.data
        self.gaps.fill(0);self.gaps[:m]=world.gaps;self.free[:]=free
        for i,(R,tangent) in enumerate(mapping.parts):
            self.rot[i][:]=R
            if tangent is not None:self.tangent[i][:]=tangent
        self.prepare_graph.launch();self.sim.blocks=self.blocks;self.sim.contact=self.contacts;self.sim.n=m
        return self.free_gap[:m]

    def recover(self,lam):
        self.lam.fill(0);self.lam[:len(lam)]=lam;self.recover_graph.launch()
        # The caller may retain an earlier trial during backtracking.
        return tuple(v.copy() for v in (self.direction,self.targets,self.forces,self.targetd))
