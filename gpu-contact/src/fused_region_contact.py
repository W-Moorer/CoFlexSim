"""Fused current surface prediction and reference compliance projection."""
import cupy as cp
import numpy as np
from direct_region_contact import DirectRegionContact

CODE=r'''
__device__ double warp_add(double x){for(int d=16;d;d/=2)x+=__shfl_down_sync(0xffffffff,x,d);return x;}
extern "C" __global__ void prepare_surface(const double*P,const double*u,const double*dd,const double*ds,
 const double*R,const double*tangent,double*pu,double*out,int nodes,int modes,int o,int warp_mode){
 int thread=blockIdx.x*blockDim.x+threadIdx.x,node=warp_mode?thread/32:thread,lane=warp_mode?threadIdx.x%32:0;
 if(node>=nodes)return;double y[3];
 for(int k=0;k<3;k++){int row=3*node+k;double a=0.,b=0.;
  for(int j=lane;j<modes;j+=warp_mode?32:1){double z=P[row*modes+j];a+=z*u[j];b+=z*dd[o+j];}
  if(warp_mode){a=warp_add(a);b=warp_add(b);}if(!lane){pu[row]=a;y[k]=b+ds[row];}
 }
 if(lane)return;for(int k=0;k<3;k++){double z=o?tangent[3*node+k]*dd[0]:0.;for(int j=0;j<3;j++)z+=R[3*k+j]*y[j];out[3*node+k]=z;}
}
extern "C" __global__ void free_contact_gap(const int*ptr,const int*col,const double*val,const double*x,
 const double*gap,double*out,int rows){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows)return;double y=0.;
 for(int e=ptr[i];e<ptr[i+1];e++)y+=val[e]*x[col[e]];out[i]=gap[i]+y;
}
'''

class FusedRegionContact(DirectRegionContact):
    def __init__(self,sim):
        super().__init__(sim);module=cp.RawModule(code=CODE)
        self.surface_kernel=module.get_function('prepare_surface');self.gap_kernel=module.get_function('free_contact_gap')
        if sim.scene.get('regional_warp_rows',False):
            from parallel_contact_rows import install_prepare
            install_prepare(self)
        if sim.scene.get('regional_bounded_entries',False):
            from bounded_contact_entries import install_body
            install_body(self)
        self.live_prepare=None
        if sim.scene.get('regional_live_surface_prepare',False):
            from live_contact_execution import LiveSurfacePreparation
            self.live_prepare=LiveSurfacePreparation(self)

    def prepare(self,world,mapping,free):
        if self.live_prepare is not None:return self.live_prepare.prepare(world,mapping,free)
        sim=self.sim;m=len(world.gaps);e=len(world.G.data)
        if self.signature!=(world.G.data.data.ptr,world.G.indptr.data.ptr,m,e):self.allocate_direct(world)
        self.free=free;G=world.G;mapped=cp.empty(sim.ns)
        for i,(b,B,(R,tangent),(rows,C,rank),(_,jr,v)) in enumerate(zip(sim.bodies,sim.dynamic,mapping.parts,self.blocks,self.contacts)):
            lo,hi=sim.soff[i:i+2];dl,dh=sim.doff[i:i+2];groups=e//3;n=len(B.X);modes=B.P.shape[1];wide=modes>=32
            pu=cp.empty(hi-lo)
            self.surface_kernel((((n*32 if wide else n)+127)//128,),(128,),(B.P,B.u,free[dl:dh],free[sim.nd+lo:sim.nd+hi],R,
                tangent if tangent is not None else cp.empty(0),pu,mapped[lo:hi],np.int32(n),np.int32(modes),np.int32(b.offset),np.int32(wide)))
            self.body_kernel(((groups+127)//128,),(128,),(G.indices,G.data,rows.indices,rows.data,np.int32(groups),np.int32(lo),np.int32(hi-lo),R,G.indptr,np.int32(m)))
            if b.reference:self.reference_kernel(((m+127)//128,),(128,),(G.indptr,G.indices,G.data,rows.indices,rows.data,tangent,pu,B.sigma,
                np.int32(lo),np.int32(hi-lo),np.int32(m),jr,v,rank))
        sim.blocks=self.blocks;sim.contact=self.contacts;sim.n=m;gap=cp.empty(m)
        self.gap_kernel(((m+127)//128,),(128,),(G.indptr,G.indices,G.data,mapped,world.gaps,gap,np.int32(m)))
        return gap
