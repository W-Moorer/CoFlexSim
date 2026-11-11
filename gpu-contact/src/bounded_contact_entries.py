"""Process live CSR entries with bounded grids, independent of reserve capacity."""
import numpy as np
import cupy as cp

CODE=r'''
extern "C" __global__ void nodes_and_coverage(const int*col,const double*value,int*seen,int*ids,int*count,
 int entries,const int*ptr,int rows,const int*lookup,int*error){
 int end=min(entries,ptr[rows]);
 for(int e=blockIdx.x*blockDim.x+threadIdx.x;e<end;e+=gridDim.x*blockDim.x)if(value[e]!=0.){
  int c=col[e];if(lookup[c]<0)atomicOr(error,8192);
  if(atomicCAS(seen+c,-1,-2)==-1){int i=atomicAdd(count,1);ids[i]=c;seen[c]=i;}
 }
}
extern "C" __global__ void body_live_rows(const int*c,const double*a,int*bc,double*ba,int groups,int offset,int n,
 const double*R,const int*ptr,int rows){
 int end=min(groups,ptr[rows]/3);
 for(int j=blockIdx.x*blockDim.x+threadIdx.x;j<end;j+=gridDim.x*blockDim.x){int col=c[3*j]-offset;
  for(int k=0;k<3;k++){
   bc[3*j+k]=(col>=0&&col<n)?col+k:k;
   ba[3*j+k]=(col>=0&&col<n)?a[3*j]*R[k]+a[3*j+1]*R[3+k]+a[3*j+2]*R[6+k]:0.;
  }
 }
}
'''

def install_working(working,device):
    kernel=cp.RawModule(code=CODE).get_function('nodes_and_coverage')
    def nodes(grid,block,args):
        index=next(i for i,buf in enumerate(working.buffers) if buf[2].data.ptr==args[1].data.ptr)
        B=device.dynamic[index]
        kernel((min(256,max(1,(int(args[5])+255)//256)),),(256,),tuple(args)+(B.window_lookup,device.geometry.error))
    working.region_k['region_nodes']=nodes
    working.check_kernel=lambda *args,**kw:None # the fused loop performs the same check

def install_body(contact):
    kernel=cp.RawModule(code=CODE).get_function('body_live_rows')
    def call(grid,block,args):kernel((min(256,max(1,(int(args[4])+255)//256)),),(256,),args)
    contact.body_kernel=call
