"""Return a compact surface-node mask instead of transferring all contact CSR."""
import numpy as np
import cupy as cp

CODE=r'''
extern "C" __global__ void mark_pairs(const int*owner,const int*master,const int*sf,const int*mf,
 int*mask,int n,int slo,int mlo){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 int s=owner[i],m=master[i];if(s<0||m<0)return;
 for(int k=0;k<3;k++){atomicExch(mask+slo+sf[3*s+k],1);atomicExch(mask+mlo+mf[3*m+k],1);}
}
extern "C" __global__ void mark_load_support(const int*ptr,const int*col,const double*val,
 int*mask,int cap,int rows){
 int end=min(cap,ptr[rows]);
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<end;i+=gridDim.x*blockDim.x)
  if(val[i]!=0.)atomicExch(mask+col[i]/3,1);
}
'''

def collect(sim):
    if not hasattr(sim,'candidate_mask'):
        mod=cp.RawModule(code=CODE)
        sim.mark_candidate_pairs=mod.get_function('mark_pairs');sim.mark_load_support=mod.get_function('mark_load_support')
        sim.candidate_mask=cp.zeros(sim.ns//3,dtype=cp.int32)
        sim.candidate_faces=[cp.asarray(b.spec['contact_faces'],dtype=cp.int32) for b in sim.bodies]
    sim.candidate_mask.fill(0)
    for part in sim.geometry.parts:
        n=len(part.owners)
        if n:
            if sim.scene.get('regional_current_pair_filter',False):
                from regional_pair_filter import mark
                mark(sim.geometry,part,sim.candidate_mask)
            else:sim.mark_candidate_pairs(((n+127)//128,),(128,),
                (part.owners,part.mids,sim.candidate_faces[part.slave],sim.candidate_faces[part.master],sim.candidate_mask,
                 np.int32(n),np.int32(sim.soff[part.slave]//3),np.int32(sim.soff[part.master]//3)))
    if hasattr(sim,'world'):
        G=sim.world.G;n=len(G.data)
        if n:sim.mark_load_support((min(256,(n+255)//256),),(256,),
            (G.indptr,G.indices,G.data,sim.candidate_mask,np.int32(n),np.int32(G.shape[0])))
    mask=cp.asnumpy(sim.candidate_mask)
    out=[]
    for lo,hi in zip(sim.soff[:-1]//3,sim.soff[1:]//3):
        ids=np.flatnonzero(mask[lo:hi]);out.append((3*ids[:,None]+np.arange(3)).ravel())
    return out
