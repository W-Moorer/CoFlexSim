"""Stable duplicate-node reduction of current contact CSR, entirely on CUDA.

Keep xyz triplets, global pressure row IDs and all candidate gaps. Rows beyond
the bounded shared sort capacity pass through unchanged; no support is dropped.
"""
import numpy as np
import cupy as cp
from types import SimpleNamespace
from device_geometry import DeviceWorld

CODE=r'''
extern "C" __global__ void merge_nodes(const int*ptr,const int*cols,const double*vals,
 int*tmpcols,double*tmpvals,int*counts,int m,int*stats){
 int r=blockIdx.x,t=threadIdx.x;if(r>=m)return;
 int start=ptr[r],n=(ptr[r+1]-start)/3;
 if(!n){if(!t)counts[r]=0;return;}
 if(n>512){for(int j=t;j<3*n;j+=blockDim.x){tmpcols[start+j]=cols[start+j];tmpvals[start+j]=vals[start+j];}
  if(!t){counts[r]=3*n;atomicAdd(stats+1,1);}return;}
 __shared__ int key[512],order[512];__shared__ double x[512],y[512],z[512];
 for(int j=t;j<512;j+=blockDim.x){key[j]=j<n?cols[start+3*j]/3:2147483647;order[j]=j;
  x[j]=j<n?vals[start+3*j]:0.;y[j]=j<n?vals[start+3*j+1]:0.;z[j]=j<n?vals[start+3*j+2]:0.;}
 __syncthreads();
 for(int size=2;size<=512;size*=2)for(int stride=size/2;stride;stride/=2){
  for(int j=t;j<512;j+=blockDim.x){int k=j^stride;if(k>j){bool larger=key[j]>key[k]||(key[j]==key[k]&&order[j]>order[k]);
   if(((j&size)==0&&larger)||((j&size)!=0&&!larger)){
    int q=key[j];key[j]=key[k];key[k]=q;q=order[j];order[j]=order[k];order[k]=q;
    double a=x[j];x[j]=x[k];x[k]=a;a=y[j];y[j]=y[k];y[k]=a;a=z[j];z[j]=z[k];z[k]=a;}}}
  __syncthreads();}
 if(!t){int used=0,j=0;while(j<n){int node=key[j];double a=x[j],b=y[j],c=z[j];j++;
   while(j<n&&key[j]==node){a+=x[j];b+=y[j];c+=z[j];j++;}
   int pos=start+3*used;for(int k=0;k<3;k++)tmpcols[pos+k]=3*node+k;
   tmpvals[pos]=a;tmpvals[pos+1]=b;tmpvals[pos+2]=c;used++;}
  counts[r]=3*used;atomicAdd(stats,3*(n-used));}
}
extern "C" __global__ void pack_merged(const int*oldptr,const int*newptr,const int*tc,const double*tv,
 int*c,double*v,int m){int r=blockIdx.x;if(r>=m)return;int n=newptr[r+1]-newptr[r];
 for(int j=threadIdx.x;j<n;j+=blockDim.x){c[newptr[r]+j]=tc[oldptr[r]+j];v[newptr[r]+j]=tv[oldptr[r]+j];}}
'''

class CompactRows:
    def __init__(self):
        mod=cp.RawModule(code=CODE);self.merge=mod.get_function('merge_nodes');self.pack=mod.get_function('pack_merged')
        self.signature=None;self.stats=cp.zeros(2,dtype=cp.int32)

    def apply(self,world):
        G=world.G;m=G.shape[0];capacity=len(G.data);signature=(G.indptr.data.ptr,G.data.data.ptr,G.indices.data.ptr,m,capacity)
        if self.signature!=signature:
            self.tmpcols=cp.empty(capacity,dtype=cp.int32);self.tmpvals=cp.empty(capacity)
            self.counts=cp.zeros(m,dtype=cp.int32);self.ptr=cp.zeros(m+1,dtype=cp.int32)
            self.cols=cp.zeros(capacity,dtype=cp.int32);self.values=cp.zeros(capacity)
            self.G=SimpleNamespace(indptr=self.ptr,indices=self.cols,data=self.values,shape=G.shape)
            self.signature=signature
        self.merge((m,),(128,),(G.indptr,G.indices,G.data,self.tmpcols,self.tmpvals,self.counts,np.int32(m),self.stats))
        cp.cumsum(self.counts,dtype=cp.int32,out=self.ptr[1:])
        self.pack((m,),(128,),(G.indptr,self.ptr,self.tmpcols,self.tmpvals,self.cols,self.values,np.int32(m)))
        return DeviceWorld(world.gaps,self.G,world.keys,world.physical_min,world.stats)

def install(geometry):
    if hasattr(geometry,'compact_rows'):return
    if getattr(geometry,'live_support_enabled',False):
        from live_contact_execution import LiveCompactRows
        geometry.compact_rows=LiveCompactRows(include_nodes=geometry.live_surface_prepare,
                                             cache_sort=geometry.live_sort_cache)
    else:geometry.compact_rows=CompactRows()
    old=geometry.enqueue
    def enqueue(x):return geometry.compact_rows.apply(old(x))
    geometry.enqueue=enqueue
