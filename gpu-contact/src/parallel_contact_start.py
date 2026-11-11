"""Stable parallel warm-start selection and minimum-gap scan.

Seed order and ties match the original ascending-row device_start. Only launch
organization changes; all candidates and previous multipliers are considered.
"""
import numpy as np
import cupy as cp

CODE=r'''
#define INF __longlong_as_double(0x7ff0000000000000LL)
extern "C" __global__ void start_rows(const double*b,const long long*keys,const long long*oldkeys,
 const double*oldw,const int*oldcount,const int*valid,int m,int reuse,
 int*prefix,double*seed,double*mins,int*argmins,int*counts){
 int t=threadIdx.x,i=blockIdx.x*blockDim.x+t;__shared__ double mb[128];__shared__ int mi[128],scan[128];
 double w=0.;if(i<m&&reuse&&valid[i])for(int z=0;z<oldcount[0];z++)if(keys[i]==oldkeys[z]&&oldw[z]>1e-8){w=oldw[z];break;}
 if(i<m)seed[i]=w;int yes=w>0.;scan[t]=yes;mb[t]=i<m?b[i]:INF;mi[t]=i<m?i:2147483647;__syncthreads();
 for(int d=1;d<128;d*=2){int x=t>=d?scan[t-d]:0;__syncthreads();scan[t]+=x;__syncthreads();}
 if(i<m)prefix[i]=scan[t]-yes;if(t==127)counts[blockIdx.x]=scan[t];
 for(int d=64;d;d/=2){if(t<d&&(mb[t+d]<mb[t]||(mb[t+d]==mb[t]&&mi[t+d]<mi[t]))){mb[t]=mb[t+d];mi[t]=mi[t+d];}__syncthreads();}
 if(!t){mins[blockIdx.x]=mb[0];argmins[blockIdx.x]=mi[0];}
}
extern "C" __global__ void start_summary(const double*mins,const int*argmins,int*counts,const double*seed,
 int*meta,int*worst,int blocks,int m,int cap){
 if(threadIdx.x)return;for(int k=0;k<7;k++)meta[k]=0;meta[0]=m;
 int chosen=-1;double best=INF;for(int k=0;k<blocks;k++)if(mins[k]<best||(mins[k]==best&&(chosen<0||argmins[k]<chosen))){best=mins[k];chosen=argmins[k];}
 worst[0]=chosen;if(!m||best>=0.){meta[4]=1;return;}
 int total=0;for(int k=0;k<blocks;k++){int n=counts[k];counts[k]=total;total+=n;}
 if(total>cap){meta[5]=5;return;}meta[1]=total;int size=total+(seed[chosen]<=0.);
 if(size>cap){meta[5]=5;return;}meta[2]=size;
}
extern "C" __global__ void start_pack(const int*prefix,const double*seed,const int*counts,const int*worst,
 const int*meta,long long*ids,double*weights,int m){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=m||meta[4]||meta[5])return;
 if(seed[i]>0.){int slot=counts[blockIdx.x]+prefix[i];ids[slot]=i;weights[slot]=seed[i];}
 if(i==worst[0]&&seed[i]<=0.){ids[meta[1]]=i;weights[meta[1]]=0.;}
}
'''

def workspace(working,m):
    blocks=max(1,(m+127)//128)
    working.start_workspace=(cp.empty(m,dtype=cp.int32),cp.empty(m),cp.empty(blocks),cp.empty(blocks,dtype=cp.int32),
                             cp.empty(blocks,dtype=cp.int32),cp.empty(1,dtype=cp.int32))

def install(working):
    mod=cp.RawModule(code=CODE);rows=mod.get_function('start_rows');summary=mod.get_function('start_summary');pack=mod.get_function('start_pack')
    def call(grid,block,args):
        b,keys,oldkeys,oldw,oldcount,valid,meta,ids,weights,m,cap,reuse=args
        prefix,seed,mins,argmins,counts,worst=working.start_workspace;nb=len(counts)
        rows((nb,),(128,),(b,keys,oldkeys,oldw,oldcount,valid,m,reuse,prefix,seed,mins,argmins,counts))
        summary((1,),(1,),(mins,argmins,counts,seed,meta,worst,np.int32(nb),m,cap))
        pack((nb,),(128,),(prefix,seed,counts,worst,meta,ids,weights,m))
    working.region_k['device_start']=call
