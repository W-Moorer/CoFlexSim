"""Exact two-dimensional dispatch of existing contact-response assembly.

Each output keeps the original summation order. The previously serial loop
over working-set columns becomes a grid dimension; QP and pressure stay intact.
"""
import numpy as np
import cupy as cp

CODE=r'''
extern "C" __global__ void residual_columns(const double*C,const int*ptr,const int*col,const double*val,
 const long long*rows,const int*meta,double*T,int p,int cap,int column_major,const int*nodes,const int*count,const int*lookup){
 int ii=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y;
 if(meta[4]||meta[5]||ii>=count[0]||j<meta[3]||j>=meta[2])return;
 int i=nodes[ii],row=rows[j],ri=lookup[i];double v=0.;
 for(int e=ptr[row];e<ptr[row+1];e++)if(val[e]!=0.){
  int cj=lookup[col[e]];
  v+=((ri>=0&&cj>=0)?C[(long long)cj*p+ri]:0.)*val[e];
 }T[(long long)ii*cap+j]=v;
}
extern "C" __global__ void modal_columns(const double*P,const double*Y,
 const int*nodes,const int*count,const int*meta,double*T,int modes,int cap){
 int ii=blockIdx.x,j=blockIdx.y,t=threadIdx.x;
 if(meta[4]||meta[5]||ii>=count[0]||j<meta[3]||j>=meta[2])return;
 int i=nodes[ii];__shared__ double sums[128];double s=0.;
 for(int k=t;k<modes;k+=blockDim.x)s+=P[(long long)i*modes+k]*Y[(long long)j*modes+k];
 sums[t]=s;__syncthreads();
 for(int d=64;d;d/=2){if(t<d)sums[t]+=sums[t+d];__syncthreads();}
 if(!t)T[(long long)ii*cap+j]+=sums[0];
}
extern "C" __global__ void modal_columns_live(const double*P,const double*Y,
 const int*nodes,const int*count,const int*meta,double*T,int modes,int cap){
 if(meta[4]||meta[5])return;int nc=meta[2]-meta[3],t=threadIdx.x;
 __shared__ double sums[128];
 for(int item=blockIdx.x;item<count[0]*nc;item+=gridDim.x){
  int ii=item/nc,j=meta[3]+item%nc,i=nodes[ii];double s=0.;
  for(int k=t;k<modes;k+=blockDim.x)s+=P[(long long)i*modes+k]*Y[(long long)j*modes+k];
  sums[t]=s;__syncthreads();
  for(int d=64;d;d/=2){if(t<d)sums[t]+=sums[t+d];__syncthreads();}
  if(!t)T[(long long)ii*cap+j]+=sums[0];__syncthreads();
 }
}
extern "C" __global__ void contact_columns(const int*ptr,const int*col,const double*val,const double*v,
 const long long*rows,const int*meta,const double*T,double*W,int cap,int add,const int*seen){
 int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y;
 if(meta[4]||meta[5]||i>=meta[0]||j<meta[3]||j>=meta[2])return;
 double s=v[i]*v[rows[j]];
 for(int e=ptr[i];e<ptr[i+1];e++)if(val[e]!=0.)s+=val[e]*T[(long long)seen[col[e]]*cap+j];
 if(add)W[(long long)i*cap+j]+=s;else W[(long long)i*cap+j]=s;
}
'''

def install(working,device):
    module=cp.RawModule(code=CODE)
    static=module.get_function('residual_columns');modal=module.get_function('modal_columns');contact=module.get_function('contact_columns')
    live=module.get_function('modal_columns_live')
    def response(grid,block,args):
        i=next(i for i,B in enumerate(device.dynamic) if B.C.data.ptr==args[0].data.ptr)
        B=device.dynamic[i];_,ptr,col,val,rows,meta,T,_,cap,_,nodes,count=args
        updated=list(args);updated[7]=np.int32(B.C.shape[0])
        static(((B.C.shape[0]+127)//128,int(cap)),(128,),tuple(updated)+(B.window_lookup,))
        modes=B.P.shape[1]
        if modes:
            working.modal_direction(((modes+127)//128,int(cap)),(128,),
                (B.regional_modal_response,B.window_lookup,ptr,col,val,rows,meta,working.modal_columns[i],np.int32(modes),cap))
            ma=(B.P,working.modal_columns[i],nodes,count,meta,T,np.int32(modes),cap)
            if device.scene.get('regional_persistent_modal',False):live((256,),(128,),ma)
            else:modal((B.C.shape[0],int(cap)),(128,),ma)
    def assemble(grid,block,args):
        cap=int(args[8]);contact((grid[0],cap),block,args)
    working.region_k['region_response']=response
    working.region_k['region_contact']=assemble
