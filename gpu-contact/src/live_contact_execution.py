"""Exact live-support execution with unchanged global pressure IDs and checks.

Only work lists are compacted. Geometry, pressure integrals, physical gaps and
all-candidate acceptance tests retain the original definitions.
"""
from types import SimpleNamespace
import numpy as np
import cupy as cp
from compact_contact_rows import CODE as COMPACT_CODE
from device_geometry import DeviceWorld
from fused_region_contact import CODE as PREPARE_CODE

LIST_CODE = r'''
extern "C" __global__ void row_flags(const int*ptr,int*flag,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n)flag[i]=ptr[i+1]>ptr[i];
}
extern "C" __global__ void pack_list(const int*flag,const int*prefix,int*ids,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n&&flag[i])ids[prefix[i]-1]=i;
}
extern "C" __global__ void mark_nodes(const int*ptr,const int*cols,int*flags,int rows){
 int end=ptr[rows]/3;
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<end;i+=gridDim.x*blockDim.x)
  atomicExch(flags+cols[3*i]/3,1);
}
'''


def compact_source(cache_sort=False):
    source = COMPACT_CODE
    source = source.replace('int*counts,int m,int*stats){',
                            'int*counts,int m,int*stats,const int*live,const int*prefix){')
    source = source.replace('int r=blockIdx.x,t=threadIdx.x;if(r>=m)return;',
                            'int t=threadIdx.x; for(int item=blockIdx.x;item<prefix[m-1];item+=gridDim.x){ int r=live[item];')
    source = source.replace('if(!n){if(!t)counts[r]=0;return;}',
                            'if(!n){if(!t)counts[r]=0;continue;}')
    source = source.replace('atomicAdd(stats+1,1);}return;}',
                            'atomicAdd(stats+1,1);}__syncthreads();continue;}')
    source = source.replace('__shared__ int key[512]',
                            'int size_limit=1;while(size_limit<n)size_limit*=2;\n __shared__ int key[512]')
    source = source.replace('j<512;', 'j<size_limit;').replace('size<=512;', 'size<=size_limit;')
    source = source.replace('counts[r]=3*used;atomicAdd(stats,3*(n-used));}\n}',
                            'counts[r]=3*used;atomicAdd(stats,3*(n-used));}\n __syncthreads();}\n}')
    source = source.replace('int*c,double*v,int m){int r=blockIdx.x;if(r>=m)return;',
                            'int*c,double*v,int m,const int*live,const int*prefix){ for(int item=blockIdx.x;item<prefix[m-1];item+=gridDim.x){int r=live[item];')
    source = source.replace('v[newptr[r]+j]=tv[oldptr[r]+j];}}',
                            'v[newptr[r]+j]=tv[oldptr[r]+j];}}}')
    if cache_sort:
        source=source.replace('const int*live,const int*prefix){\n int t=',
            'const int*live,const int*prefix,int*cached_n,int*cached_nodes,int*cached_order){\n int t=',1)
        marker='for(int j=t;j<size_limit;j+=blockDim.x){key[j]='
        prefix='''__shared__ int changed;
 if(!t)changed=cached_n[r]!=n;__syncthreads();
 for(int j=t;j<n;j+=blockDim.x)if(cached_nodes[r*512+j]!=cols[start+3*j]/3)atomicExch(&changed,1);
 __syncthreads();
 if(!changed){
  for(int j=t;j<n;j+=blockDim.x){int pos=cached_order[r*512+j];key[j]=cols[start+3*pos]/3;order[j]=pos;
   x[j]=vals[start+3*pos];y[j]=vals[start+3*pos+1];z[j]=vals[start+3*pos+2];}
  __syncthreads();if(!t)atomicAdd(stats+2,1);
 }else{
 '''
        assert source.count(marker)==1
        source=source.replace(marker,prefix+marker)
        marker2='if(!t){int used=0,j=0;'
        suffix='''for(int j=t;j<n;j+=blockDim.x){cached_nodes[r*512+j]=cols[start+3*j]/3;cached_order[r*512+j]=order[j];}
 if(!t){cached_n[r]=n;atomicAdd(stats+3,1);}__syncthreads();}
 '''
        assert source.count(marker2)==1
        source=source.replace(marker2,suffix+marker2)
    return source + LIST_CODE


class LiveCompactRows:
    def __init__(self, include_nodes=False, cache_sort=False):
        module = cp.RawModule(code=compact_source(cache_sort))
        self.k = {n: module.get_function(n) for n in
                  ('merge_nodes', 'pack_merged', 'row_flags', 'pack_list', 'mark_nodes')}
        self.signature = None
        self.stats = cp.zeros(4, dtype=cp.int32)
        self.include_nodes = include_nodes
        self.cache_sort = cache_sort

    def apply(self, world):
        G = world.G
        m, n = G.shape
        signature = (G.indptr.data.ptr, G.data.data.ptr, G.indices.data.ptr, m, len(G.data))
        if signature != self.signature:
            cap = len(G.data)
            self.tmpcols = cp.empty(cap, dtype=cp.int32)
            self.tmpvals = cp.empty(cap)
            self.counts = cp.zeros(m, dtype=cp.int32)
            self.ptr = cp.zeros(m+1, dtype=cp.int32)
            self.cols = cp.empty(cap, dtype=cp.int32)
            self.values = cp.empty(cap)
            self.flags = cp.empty(m, dtype=cp.int32)
            self.prefix = cp.empty(m, dtype=cp.int32)
            self.live = cp.empty(m, dtype=cp.int32)
            if self.cache_sort:
                self.cached_n=cp.full(m,-1,dtype=cp.int32)
                self.cached_nodes=cp.full((m,512),-1,dtype=cp.int32)
                self.cached_order=cp.empty((m,512),dtype=cp.int32)
            self.G = SimpleNamespace(indptr=self.ptr, indices=self.cols,
                                     data=self.values, shape=G.shape)
            if self.include_nodes:
                self.node_flags = cp.zeros(n//3, dtype=cp.int32)
                self.node_prefix = cp.empty(n//3, dtype=cp.int32)
                self.node_ids = cp.empty(n//3, dtype=cp.int32)
            self.signature = signature
        self.counts.fill(0)
        if m:
            self.k['row_flags'](((m+127)//128,), (128,), (G.indptr,self.flags,np.int32(m)))
            cp.cumsum(self.flags, dtype=cp.int32, out=self.prefix)
            self.k['pack_list'](((m+127)//128,), (128,), (self.flags,self.prefix,self.live,np.int32(m)))
            self.k['merge_nodes']((min(256,m),), (128,),
                (G.indptr,G.indices,G.data,self.tmpcols,self.tmpvals,self.counts,np.int32(m),self.stats,self.live,self.prefix)+
                ((self.cached_n,self.cached_nodes,self.cached_order) if self.cache_sort else ()))
            cp.cumsum(self.counts, dtype=cp.int32, out=self.ptr[1:])
            self.k['pack_merged']((min(256,m),), (128,),
                (G.indptr,self.ptr,self.tmpcols,self.tmpvals,self.cols,self.values,np.int32(m),self.live,self.prefix))
        out = DeviceWorld(world.gaps,self.G,world.keys,world.physical_min,world.stats)
        out.live_rows = (self.live, self.prefix[-1:])
        if self.include_nodes:
            nodes = len(self.node_flags)
            self.node_flags.fill(0)
            self.k['mark_nodes']((256,), (128,), (self.ptr,self.cols,self.node_flags,np.int32(m)))
            cp.cumsum(self.node_flags, dtype=cp.int32, out=self.node_prefix)
            self.k['pack_list'](((nodes+127)//128,), (128,),
                (self.node_flags,self.node_prefix,self.node_ids,np.int32(nodes)))
            out.supported_nodes = (self.node_ids,self.node_prefix[-1:])
        return out


def prepare_source():
    source = PREPARE_CODE
    source = source.replace('int nodes,int modes,int o,int warp_mode){',
                            'int nodes,int modes,int o,int warp_mode,const int*ids,const int*count,int offset){')
    source = source.replace('node=warp_mode?thread/32:thread,lane=',
                            'item=warp_mode?thread/32:thread,lane=')
    source = source.replace('if(node>=nodes)return;double y[3];',
                            'for(;item<count[0];item+=gridDim.x*blockDim.x/(warp_mode?32:1)){int node=ids[item]-offset;if(node<0||node>=nodes)continue;double y[3];')
    source = source.replace('if(lane)return;', 'if(lane)continue;')
    source = source.replace('out[3*node+k]=z;}\n}', 'out[3*node+k]=z;}}\n}')
    return source


class LiveSurfacePreparation:
    def __init__(self, contact):
        self.contact = contact
        contact.surface_kernel = cp.RawModule(code=prepare_source()).get_function('prepare_surface')

    def prepare(self, world, mapping, free):
        contact = self.contact
        sim = contact.sim
        if not hasattr(world, 'supported_nodes'):
            raise RuntimeError('Live preparation requires the current CSR node list')
        m, e = len(world.gaps), len(world.G.data)
        if contact.signature != (world.G.data.data.ptr,world.G.indptr.data.ptr,m,e):
            contact.allocate_direct(world)
        contact.free = free
        G = world.G
        mapped = cp.empty(sim.ns)
        ids, count = world.supported_nodes
        for i,(b,B,(R,tangent),(rows,C,rank),(_,jr,v)) in enumerate(zip(
                sim.bodies,sim.dynamic,mapping.parts,contact.blocks,contact.contacts)):
            lo,hi = sim.soff[i:i+2]
            dl,dh = sim.doff[i:i+2]
            groups,n,modes = e//3,len(B.X),B.P.shape[1]
            wide = modes >= 32
            pu = cp.empty(hi-lo)
            # Foreign-body zero entries in body_rows point to this finite dummy.
            pu[:3].fill(0.)
            contact.surface_kernel((256,), (128,),
                (B.P,B.u,free[dl:dh],free[sim.nd+lo:sim.nd+hi],R,
                 tangent if tangent is not None else cp.empty(0),pu,mapped[lo:hi],
                 np.int32(n),np.int32(modes),np.int32(b.offset),np.int32(wide),ids,count,np.int32(lo//3)))
            contact.body_kernel(((groups+127)//128,), (128,),
                (G.indices,G.data,rows.indices,rows.data,np.int32(groups),np.int32(lo),np.int32(hi-lo),R,G.indptr,np.int32(m)))
            if b.reference:
                contact.reference_kernel(((m+127)//128,), (128,),
                    (G.indptr,G.indices,G.data,rows.indices,rows.data,tangent,pu,B.sigma,
                     np.int32(lo),np.int32(hi-lo),np.int32(m),jr,v,rank))
        sim.blocks,sim.contact,sim.n = contact.blocks,contact.contacts,m
        gap = cp.empty(m)
        contact.gap_kernel(((m+127)//128,), (128,),
            (G.indptr,G.indices,G.data,mapped,world.gaps,gap,np.int32(m)))
        return gap
