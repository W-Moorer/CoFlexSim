"""Warp-cooperative CSR products, unchanged FP64 contact virtual work.

Only summation scheduling changes. Strict trajectory checks are required since
parallel reductions need not reproduce serial floating-point addition exactly.
"""
import cupy as cp

CODE=r'''
__device__ double sum_warp(double x){for(int d=16;d;d/=2)x+=__shfl_down_sync(0xffffffff,x,d);return x;}
extern "C" __global__ void reference_rows_warp(const int*p,const int*c,const double*a,const int*bc,const double*ba,
 const double*tangent,const double*pu,const double*sigma,int offset,int n,int rows,double*jr,double*v,double*rank){
 int r=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;if(r>=rows)return;
 double d=0.,u=0.;for(int e=p[r]+lane;e<p[r+1];e+=32){int k=c[e]-offset;if(k>=0&&k<n)d+=a[e]*tangent[k];u+=ba[e]*pu[bc[e]];}
 d=sum_warp(d);u=sum_warp(u);if(!lane){jr[r]=d;v[r]=d-u;rank[r]=(d-u)/sqrt(sigma[0]);}
}
extern "C" __global__ void gap_warp(const int*p,const int*c,const double*a,const double*x,const double*gap,double*out,int rows){
 int r=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;if(r>=rows)return;
 double s=0.;for(int e=p[r]+lane;e<p[r+1];e+=32)s+=a[e]*x[c[e]];s=sum_warp(s);if(!lane)out[r]=gap[r]+s;
}
extern "C" __global__ void contact_columns_warp(const int*ptr,const int*col,const double*val,const double*v,
 const long long*rows,const int*meta,const double*T,double*W,int cap,int add,const int*seen){
 int i=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32,j=blockIdx.y;
 if(meta[4]||meta[5]||i>=meta[0]||j<meta[3]||j>=meta[2])return;
 double s=0.;for(int e=ptr[i]+lane;e<ptr[i+1];e+=32)if(val[e]!=0.)s+=val[e]*T[(long long)seen[col[e]]*cap+j];
 s=sum_warp(s);if(!lane){s+=v[i]*v[rows[j]];if(add)W[(long long)i*cap+j]+=s;else W[(long long)i*cap+j]=s;}
}
'''

def install_prepare(contact):
    mod=cp.RawModule(code=CODE);reference=mod.get_function('reference_rows_warp');gap=mod.get_function('gap_warp')
    def ref_call(grid,block,args):
        rows=int(args[10]);reference(((rows+3)//4,),(128,),args)
    def gap_call(grid,block,args):
        rows=int(args[-1]);gap(((rows+3)//4,),(128,),args)
    contact.reference_kernel=ref_call;contact.gap_kernel=gap_call

def install_working(working):
    kernel=cp.RawModule(code=CODE).get_function('contact_columns_warp')
    def call(grid,block,args):kernel(((working.mcap+3)//4,int(args[8])),(128,),args)
    working.region_k['region_contact']=call
