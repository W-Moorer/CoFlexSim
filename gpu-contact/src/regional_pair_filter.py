"""Conservative current-geometry filter for operator residency, not contact law.

The broad BVH list is unchanged. An unrecovered vertex can differ by at most
the certified envelope per coordinate, so two AABBs need a 2*envelope margin.
The same predicate drives region requests and the acceptance coverage check.
"""
import numpy as np
import cupy as cp

CODE=r'''
__device__ bool near_pair(const double*x,const int*sf,const int*mf,int s,int m,int slo,int mlo,double distance){
 if(s<0||m<0)return false;
 for(int k=0;k<3;k++){
  double sl=x[3*(slo+sf[3*s])+k],sh=sl,ml=x[3*(mlo+mf[3*m])+k],mh=ml;
  for(int j=1;j<3;j++){double a=x[3*(slo+sf[3*s+j])+k],b=x[3*(mlo+mf[3*m+j])+k];
   sl=fmin(sl,a);sh=fmax(sh,a);ml=fmin(ml,b);mh=fmax(mh,b);}
  if(mh<sl-distance||ml>sh+distance)return false;
 }return true;
}
extern "C" __global__ void mark_near(const int*owners,const int*mids,const int*sf,const int*mf,
 const double*x,int*mask,int n,int slo,int mlo,double distance){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int s=owners[i],m=mids[i];
 if(!near_pair(x,sf,mf,s,m,slo,mlo,distance))return;
 for(int k=0;k<3;k++){atomicExch(mask+slo+sf[3*s+k],1);atomicExch(mask+mlo+mf[3*m+k],1);}
}
extern "C" __global__ void check_near(const int*owners,const int*mids,const int*sf,const int*mf,
 const double*x,const int*sr,const int*mr,int*error,int n,int slo,int mlo,double distance){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int s=owners[i],m=mids[i];
 if(near_pair(x,sf,mf,s,m,slo,mlo,distance)&&(!sr[s]||!mr[m]))atomicOr(error,8192);
}
'''

def prepare(geometry):
    if not hasattr(geometry,'near_mark'):
        mod=cp.RawModule(code=CODE)
        geometry.near_mark=mod.get_function('mark_near');geometry.near_check=mod.get_function('check_near')

def args(g,p,x):
    # Pad is expanded only by the existing certified-bound failure/retry path.
    return (p.owners,p.mids,g.faces[p.slave],g.faces[p.master],x),(
        np.int32(len(p.owners)),np.int32(g.offsets[p.slave]),np.int32(g.offsets[p.master]),np.float64(g.radius+2*g.pad))

def check(g,p,x,ready):
    aa,bb=args(g,p,x)
    g.near_check(((len(p.owners)+127)//128,),(128,),aa+(ready[p.slave],ready[p.master],g.error)+bb)

def mark(g,p,mask):
    aa,bb=args(g,p,g.query_x)
    g.near_mark(((len(p.owners)+127)//128,),(128,),aa+(mask,)+bb)
