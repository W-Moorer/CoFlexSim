"""Motion-certified BVH cache refresh on device, host only for capacity exits."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import cupy as cp
from cached_normal_contact import CachedNormalContact

EXTRA=r'''
extern "C" __global__ void need_refresh(const double*motion,const int*pairs,int n,double skin,int*state){
 if(threadIdx.x)return;state[0]=0;state[1]=0;
 for(int i=0;i<n;i++)if(motion[pairs[2*i]]+motion[pairs[2*i+1]]>skin)state[0]=1;
}
extern "C" __global__ void pack_query(const int*raw,const int*counts,const int*prefix,int n,int slots,
 int*owners,int*mids,int capacity,int*state,const int*overflow){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(!state[0])return;
 if(overflow[0]||prefix[n-1]>capacity){if(!i)atomicOr(state+1,overflow[0]==2?2:1);return;}
 if(i<capacity&&i>=prefix[n-1]){owners[i]=-1;mids[i]=-1;}
 if(i>=n)return;int start=i?prefix[i-1]:0,count=counts[i];
 // Traversal order is not master-index order. Rank within this face produces
 // the same lexicographic owner/master organization as the host reference.
 for(int j=0;j<count;j++){int mid=raw[i*slots+j],rank=0;for(int k=0;k<count;k++)rank+=raw[i*slots+k]<mid;
  owners[start+rank]=i;mids[start+rank]=mid;}
}
extern "C" __global__ void accept_anchor(const double*x,double*anchor,int n,const int*state,long long*count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(!state[0]||state[1])return;
 if(i<n)anchor[i]=x[i];if(!i)count[0]++;
}
extern "C" __global__ void cache_error(const int*state,int*error){
 if(!threadIdx.x&&state[1])atomicOr(error,state[1]&2?64:16);
}
extern "C" __global__ void reset_overflow(int*out,const int*state){if(!threadIdx.x&&state[0])out[0]=0;}
extern "C" __global__ void force_query(int*force,int*state){if(!threadIdx.x&&force[0]){state[0]=1;force[0]=0;}}
'''

class DeviceCachedNormal(CachedNormalContact):
    def __init__(self,bodies,scene,skin=None):
        super().__init__(bodies,scene,skin)
        source=Path(__file__).with_name('bvh.cu').read_text()
        for name in ('triangle_boxes','leaf_boxes','inner_boxes','query_boxes'):
            start=source.index('void '+name+'(');end=source.index('){',start)
            source=source[:end]+',const int*state){\n if(!state[0])return;'+source[end+2:]
        self.native_envelope=bool(scene.get('regional_native_runtime',False))
        self.query_radius_device=cp.asarray([self.radius+self.skin],dtype=cp.float64)
        self.force_query_device=cp.zeros(1,dtype=cp.int32)
        if self.native_envelope:
            marker='int n,double radius,int cap,int*out,int*counts,int*overflow,const int*state){'
            assert source.count(marker)==1
            source=source.replace(marker,marker.replace('double radius','const double* radius_device')+'\n double radius=radius_device[0];')
        module=cp.RawModule(code=source+EXTRA)
        self.dk={n:module.get_function(n) for n in ('triangle_boxes','leaf_boxes','inner_boxes','query_boxes','need_refresh','pack_query','accept_anchor','cache_error','reset_overflow','force_query')}
        self.cache_state=cp.zeros(2,dtype=cp.int32);self.device_refreshes=cp.zeros(1,dtype=cp.int64);self.device_queries=[]
        self.query_slots=int(scene.get('regional_initial_query_slots',128))
        if self.query_slots<1 or self.query_slots>4096:raise ValueError('Query slots outside supported range')
        self.exact_capacity_growth=bool(scene.get('regional_exact_capacity_growth',False))
        self.statistics['device_candidate_cache_refresh']=True

    def refresh(self,x):
        was_ready=self.ready
        needed=self.query_slots
        if self.exact_capacity_growth and self.device_queries:
            # A residual-envelope update is not a query-buffer overflow. Only
            # grow if the preceding device query actually exceeded its slots.
            for q in self.device_queries:
                flag=int(cp.asnumpy(q.overflow)[0])
                if flag==2:raise RuntimeError('BVH traversal stack overflow')
                if flag==1:
                    needed=max(needed,int(cp.max(q.counts)))
            needed=min(4096,max(self.query_slots,1 << (max(1,needed)-1).bit_length()))
            if any(int(cp.asnumpy(q.overflow)[0])==1 and int(cp.max(q.counts))>needed for q in self.device_queries):
                raise RuntimeError('BVH query needs more than 4096 per-face slots; no truncation')
        changed=super().refresh(x)
        # A repeated host refresh is an exceptional output/face-capacity exit.
        # Re-capture all pointers if device query scratch must be rebuilt.
        resize_query=(not self.device_queries or changed or
                      (needed>self.query_slots if self.exact_capacity_growth else was_ready))
        if resize_query:
            if self.exact_capacity_growth:self.query_slots=needed
            elif was_ready:self.query_slots=min(4096,2*self.query_slots)
            self.device_queries=[]
            for p in self.parts:
                n=len(self.faces[p.slave]);slots=min(self.query_slots,max(1,len(self.faces[p.master])))
                self.device_queries.append(SimpleNamespace(lo=cp.empty((n,3)),hi=cp.empty((n,3)),raw=cp.empty((n,slots),dtype=cp.int32),
                    counts=cp.zeros(n,dtype=cp.int32),prefix=cp.zeros(n,dtype=cp.int32),overflow=cp.zeros(1,dtype=cp.int32),slots=slots))
            return True
        return changed

    def enqueue(self,x):
        cp.subtract(x,self.anchor,out=self.delta);cp.abs(self.delta,out=self.delta)
        for i in range(len(self.bodies)):cp.max(self.delta[self.offsets[i]:self.offsets[i+1]],out=self.motion[i:i+1].reshape(()))
        self.dk['need_refresh']((1,),(1,),(self.motion,self.pair_array,np.int32(len(self.pairs)),np.float64(self.skin),self.cache_state))
        if self.native_envelope:self.dk['force_query']((1,),(1,),(self.force_query_device,self.cache_state))
        for master in sorted(set(p[1] for p in self.pairs)):
            tree=self.trees[master];mx=x[self.offsets[master]:self.offsets[master+1]];n=len(tree.faces)
            self.launch(self.dk['triangle_boxes'],n,(mx,tree.faces,tree.tlo,tree.thi,np.int32(n),self.cache_state))
            self.launch(self.dk['leaf_boxes'],len(tree.leaves),(tree.nodes,tree.order,tree.leaves,np.int32(len(tree.leaves)),tree.tlo,tree.thi,tree.lo,tree.hi,self.cache_state))
            for ids in tree.levels:self.launch(self.dk['inner_boxes'],len(ids),(tree.nodes,ids,np.int32(len(ids)),tree.lo,tree.hi,self.cache_state))
        for p,scratch in zip(self.parts,self.device_queries):
            n=len(self.faces[p.slave]);tree=self.trees[p.master];sx=x[self.offsets[p.slave]:self.offsets[p.slave+1]]
            self.launch(self.dk['triangle_boxes'],n,(sx,self.faces[p.slave],scratch.lo,scratch.hi,np.int32(n),self.cache_state))
            self.dk['reset_overflow']((1,),(1,),(scratch.overflow,self.cache_state))
            self.launch(self.dk['query_boxes'],n,(tree.nodes,tree.order,tree.lo,tree.hi,tree.tlo,tree.thi,scratch.lo,scratch.hi,
                np.int32(n),self.query_radius_device if self.native_envelope else np.float64(self.radius+self.skin),np.int32(scratch.slots),scratch.raw,scratch.counts,scratch.overflow,self.cache_state))
            cp.cumsum(scratch.counts,dtype=cp.int32,out=scratch.prefix)
            self.launch(self.dk['pack_query'],max(n,len(p.owners)),(scratch.raw,scratch.counts,scratch.prefix,np.int32(n),np.int32(scratch.slots),p.owners,p.mids,np.int32(len(p.owners)),self.cache_state,scratch.overflow))
        self.launch(self.dk['accept_anchor'],x.size,(x,self.anchor,np.int32(x.size),self.cache_state,self.device_refreshes))
        world=super().enqueue(x)
        self.dk['cache_error']((1,),(1,),(self.cache_state,self.error))
        return world
