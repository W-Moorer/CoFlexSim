"""Current CSR contact maps without capacity-wide staging copies.

All rows originate in the current whole-body surface geometry. Surface force
recovery remains complete; modal loads use only nonzero current force entries.
"""
from types import SimpleNamespace
import numpy as np
import cupy as cp
from resident_contact import ResidentContact,CODE as RESPONSE_CODE
from csr_actions import CODE as ACTION_CODE

SPARSE=r'''
extern "C" __global__ void sparse_modal(const double*P,const double*f,const int*ids,const int*count,double*out,int modes){
 int r=blockIdx.x,t=threadIdx.x;double sum=0.;for(int j=t;j<count[0];j+=blockDim.x){int i=ids[j];sum+=P[(long long)i*modes+r]*f[i];}
 __shared__ double work[128];work[t]=sum;__syncthreads();
 for(int s=64;s;s/=2){if(t<s)work[t]+=work[t+s];__syncthreads();}if(!t)out[r]=work[0];
}
'''

class DirectRegionContact(ResidentContact):
    def __init__(self,sim):
        super().__init__(sim);self.signature=None
        source=ACTION_CODE.replace('int groups,int offset,int n,const double*R){','int groups,int offset,int n,const double*R,const int*ptr,int rows){')
        source=source.replace('if(j>=groups)return;','if(j>=groups||3*j>=ptr[rows])return;')
        mod=cp.RawModule(code=source+SPARSE)
        self.body_kernel=mod.get_function('body_rows');self.reference_kernel=mod.get_function('reference_rows');self.sparse_modal=mod.get_function('sparse_modal')

    def allocate_direct(self,world):
        sim=self.sim;m=len(world.gaps);e=len(world.G.data);self.blocks=[];self.contacts=[];self.force_storage=[]
        for i,(b,B) in enumerate(zip(sim.bodies,sim.dynamic)):
            p=B.P.shape[0];rows=SimpleNamespace(indptr=world.G.indptr,indices=cp.empty(e,dtype=cp.int32),data=cp.empty(e),shape=(m,p))
            jr=cp.zeros(m);v=cp.zeros(m);rank=cp.zeros(m)
            self.blocks.append((rows,B.C,rank));self.contacts.append((rows,jr,v))
            self.force_storage.append((cp.empty(p,dtype=cp.int32),cp.zeros(1,dtype=cp.int32)))
        self.signature=(world.G.data.data.ptr,world.G.indptr.data.ptr,m,e);self.rebuilds+=1

    def prepare(self,world,mapping,free):
        sim=self.sim;m=len(world.gaps);e=len(world.G.data)
        if self.signature!=(world.G.data.data.ptr,world.G.indptr.data.ptr,m,e):self.allocate_direct(world)
        self.free=free;G=world.G
        for i,(b,B,(R,tangent),(rows,C,rank),(_,jr,v)) in enumerate(zip(sim.bodies,sim.dynamic,mapping.parts,self.blocks,self.contacts)):
            lo,hi=sim.soff[i:i+2];groups=e//3
            self.body_kernel(((groups+127)//128,),(128,),(G.indices,G.data,rows.indices,rows.data,np.int32(groups),np.int32(lo),np.int32(hi-lo),R,G.indptr,np.int32(m)))
            if b.reference:
                pu=self.mv(B.P,B.u)
                self.reference_kernel(((m+127)//128,),(128,),(G.indptr,G.indices,G.data,rows.indices,rows.data,tangent,pu,B.sigma,
                    np.int32(lo),np.int32(hi-lo),np.int32(m),jr,v,rank))
        sim.blocks=self.blocks;sim.contact=self.contacts;sim.n=m
        return world.gaps+sim.actions.action(G,mapping.apply(free))

    def recover(self,lam):
        sim=self.sim;direction=self.free.copy();targets=cp.zeros(sim.ns);forces=cp.zeros(sim.ns);targetd=cp.zeros(sim.nd)
        for i,(b,B,(rows,jr,v),(ids,count)) in enumerate(zip(sim.bodies,sim.dynamic,self.contacts,self.force_storage)):
            sl,sh=sim.soff[i:i+2];dl,dh=sim.doff[i:i+2];n=int(sh-sl)
            f=sim.actions.adjoint(rows,lam);count.fill(0)
            self.pack(((n+127)//128,),(128,),(f,ids,count,np.int32(n)))
            modes=B.P.shape[1];fm=cp.empty(modes)
            if modes:self.sparse_modal((modes,),(128,),(B.P,f,ids,count,fm,np.int32(modes)))
            y=self.mv(B.inverse,fm);forces[sl:sh]=f;targetd[dl+b.offset:dh]=fm
            if b.reference:
                theta=cp.sum(v*lam)/B.sigma;direction[dl]+=theta;y=y-B.u*theta;targetd[dl]=cp.sum(jr*lam)
            direction[dl+b.offset:dh]+=y
            if B.R is not None:self.response(((n+127)//128,),(128,),(B.R,f,ids,count,targets[sl:sh],np.int32(n)))
        return direction,targets,forces,targetd
