"""Conservative candidate reuse and fixed-row FP64 contact for device graphs.

The pressure discretization is copied from the frozen normal-contact kernels.
Only AABB candidate organization and storage change. Clipping and current
normals/areas are recomputed; no stale physical geometry is accepted.
"""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import cupy as cp
from fused_normal_contact import FusedNormalContact
from device_geometry import DeviceWorld
from consistent_mortar_contact import ContactGeometryError

FILTER='''
 for(int axis=0;axis<3;axis++){
  double sl=s[0][axis],sh=sl,ml=m[0][axis],mh=ml;
  for(int j=1;j<3;j++){sl=fmin(sl,s[j][axis]);sh=fmax(sh,s[j][axis]);ml=fmin(ml,m[j][axis]);mh=fmax(mh,m[j][axis]);}
  if(mh<sl-radius||ml>sh+radius)return;
 }
'''
EXTRA=r'''
extern "C" __global__ void cache_valid(const double*motion,const int*pairs,int n,double skin,int*error){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&motion[pairs[2*i]]+motion[pairs[2*i+1]]>skin)atomicOr(error,16);
}
extern "C" __global__ void fixed_rows(const double*area,const double*integral,const int*counts,
 const int*nodes,double*gaps,int*allcounts,int*valid,int n,int offset,double radius){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int node=nodes[i],r=i+offset;bool yes=area[node]>1e-24;
 gaps[r]=yes?integral[node]/area[node]:radius;allcounts[r]=yes?counts[node]*18:0;valid[r]=yes;
}
'''

class CachedNormalContact(FusedNormalContact):
    def __init__(self,bodies,scene,skin=None):
        super().__init__(bodies,scene)
        self.bounded_entries=bool(scene.get('regional_bounded_entries',False))
        self.pair_capacity_factor=float(scene.get('regional_pair_capacity_factor',1.25))
        if not 1<=self.pair_capacity_factor<=4:raise ValueError('Candidate capacity factor must be in [1,4]')
        self.skin=self.radius*.5 if skin is None else float(skin)
        source=Path(__file__).with_name('normal_contact.cu').read_text()
        marker='auto outward=read_point(normals,fs[3*face+vertex]);'
        assert source.count(marker)==1
        source=source.replace(marker,FILTER+marker)
        marker2='V<double,3>s[3],m[3];'
        assert source.count(marker2)==1
        source=source.replace(marker2,'if(face<0||master<0)return;\n '+marker2)
        marker3='auto patch=evaluate(f,embedded,m,band,false,radius);'
        assert source.count(marker3)==1
        source=source.replace(marker3,'''bool same_polygon=true;
 for(int j=0;j<guard.count;j++)same_polygon=same_polygon&&guard.vg[j]>=-radius&&guard.vg[j]<=band;
 auto patch=same_polygon?guard:evaluate(f,embedded,m,band,false,radius);''')
        module=cp.RawModule(code=Path(__file__).with_name('overlap.cu').read_text()+source+EXTRA,options=('--std=c++11',))
        self.k={n:module.get_function(n) for n in ('nodal_normals','normal_patches','check_normal_coverage','pack_normal_rows','cache_valid','fixed_rows')}
        self.ready=False;self.version=0
        self.statistics.update(candidate_cache_rebuilds=0,online_geometry='cached_current_normal_median',clipping_topology_reused=False,
            identical_guard_pressure_integrals_reused=True)

    def refresh(self,x):
        # Host management happens only on allocation or a proven motion-bound exit.
        for master in sorted(set(p[1] for p in self.pairs)):
            self.trees[master].refit(x[self.offsets[master]:self.offsets[master+1]])
        queries=[]
        for slave,master in self.pairs:
            queries.append(self.trees[master].query(x[self.offsets[slave]:self.offsets[slave+1]],self.faces[slave],self.radius+self.skin))
        if self.ready and all(len(owners)<=len(p.owners) for (owners,mids),p in zip(queries,self.parts)):
            self.anchor[:]=x
            for (owners,mids),p in zip(queries,self.parts):
                p.owners.fill(-1);p.mids.fill(-1);p.owners[:len(owners)]=owners;p.mids[:len(mids)]=mids
            self.statistics['candidate_cache_rebuilds']+=1
            self.statistics['candidate_cache_inplace_updates']=self.statistics.get('candidate_cache_inplace_updates',0)+1
            return False
        self.anchor=x.copy();self.motion=cp.zeros(len(self.bodies));self.delta=cp.empty_like(x)
        self.pair_array=cp.asarray(self.pairs,dtype=cp.int32)
        self.parts=[];keys=[];nr=0;capacity=0
        for (slave,master),(owners,mids) in zip(self.pairs,queries):
            sm=x[self.offsets[slave]:self.offsets[slave+1]];fs=self.faces[slave]
            reserve=self.pair_capacity_factor
            cap=max(64,((int(len(owners)*reserve)+255)//256)*256)
            owners=cp.pad(owners,(0,cap-len(owners)),constant_values=-1);mids=cp.pad(mids,(0,cap-len(mids)),constant_values=-1)
            nodes=cp.unique(fs).astype(cp.int32);n=len(nodes);pieces=6*len(owners)
            lut=cp.full(len(sm),-1,dtype=cp.int32);lut[nodes]=cp.arange(n,dtype=cp.int32)+nr
            p=SimpleNamespace(slave=slave,master=master,owners=owners,mids=mids,nodes=nodes,offset=nr,lut=lut,
                area=cp.zeros(len(sm)),integral=cp.zeros(len(sm)),counts=cp.zeros(len(sm),dtype=cp.int32),
                covered=cp.zeros(12*len(fs)),reference=cp.zeros(6*len(fs)),values=cp.empty((pieces,18)),
                piece_nodes=cp.empty((pieces,6),dtype=cp.int32),rows=cp.empty(pieces,dtype=cp.int32))
            self.parts.append(p);nr+=n;capacity+=pieces*18
            keys.append(((slave*len(self.bodies)+master)*self.maxnodes+nodes).astype(cp.int64))
        self.normals={i:cp.zeros((self.offsets[i+1]-self.offsets[i],3)) for i in sorted(set(p[0] for p in self.pairs))}
        self.gaps=cp.zeros(nr);self.valid=cp.zeros(nr,dtype=cp.int32);self.counts=cp.zeros(nr,dtype=cp.int32)
        self.ptr=cp.zeros(nr+1,dtype=cp.int32);self.cursor=cp.zeros(nr,dtype=cp.int32)
        self.cols=cp.zeros(capacity,dtype=cp.int32);self.values=cp.zeros(capacity)
        self.keys=cp.concatenate(keys);self.minimum=cp.zeros(1);self.error=cp.zeros(1,dtype=cp.int32)
        self.G=SimpleNamespace(indptr=self.ptr,indices=self.cols,data=self.values,shape=(nr,self.size))
        self.world=DeviceWorld(self.gaps,self.G,self.keys,self.minimum.reshape(()),self.statistics)
        self.ready=True;self.version+=1;self.statistics['candidate_cache_rebuilds']+=1
        self.statistics['cached_pairs']=sum(len(p.owners) for p in self.parts)
        return True

    def enqueue(self,x):
        self.error.fill(0);self.minimum.fill(0);self.cursor.fill(0)
        if not self.bounded_entries:self.values.fill(0);self.cols.fill(0)
        cp.subtract(x,self.anchor,out=self.delta);cp.abs(self.delta,out=self.delta)
        for i in range(len(self.bodies)):
            cp.max(self.delta[self.offsets[i]:self.offsets[i+1]],out=self.motion[i:i+1].reshape(()))
        self.launch(self.k['cache_valid'],len(self.pairs),(self.motion,self.pair_array,np.int32(len(self.pairs)),np.float64(self.skin),self.error))
        for i,n in self.normals.items():
            n.fill(0);fs=self.faces[i]
            self.launch(self.k['nodal_normals'],len(fs),(x[self.offsets[i]:self.offsets[i+1]],fs,n,np.int32(len(fs))))
        for p in self.parts:
            slave,master=p.slave,p.master;fs,fm=self.faces[slave],self.faces[master]
            sm=x[self.offsets[slave]:self.offsets[slave+1]];mm=x[self.offsets[master]:self.offsets[master+1]]
            for a in (p.area,p.integral,p.counts,p.covered,p.reference):a.fill(0)
            self.launch(self.k['normal_patches'],len(p.rows),(sm,mm,fs,fm,p.owners,p.mids,self.normals[slave],np.int32(len(p.owners)),
                np.float64(self.radius),np.float64(self.band),p.area,p.integral,p.counts,p.covered,p.reference,np.int32(len(fs)),
                p.values,p.piece_nodes,p.rows,self.minimum,self.error))
            self.launch(self.k['check_normal_coverage'],6*len(fs),(p.covered,p.reference,np.int32(len(fs)),self.error))
            self.launch(self.k['fixed_rows'],len(p.nodes),(p.area,p.integral,p.counts,p.nodes,self.gaps,self.counts,self.valid,
                np.int32(len(p.nodes)),np.int32(p.offset),np.float64(self.radius)))
        cp.cumsum(self.counts,dtype=cp.int32,out=self.ptr[1:])
        for p in self.parts:
            self.launch(self.k['pack_normal_rows'],len(p.rows),(p.rows,p.piece_nodes,p.values,p.area,p.lut,self.ptr,self.cursor,
                self.cols,self.values,np.int32(len(p.rows)),np.int32(self.offsets[p.slave]),np.int32(self.offsets[p.master])))
        return self.world

    def build(self,x,derivatives=True):
        if not self.ready:self.refresh(x)
        world=self.enqueue(x);code=int(self.error[0])
        if code&16:
            self.refresh(x);world=self.enqueue(x);code=int(self.error[0])
        if code:raise ContactGeometryError('Cached normal geometry flags '+str(code))
        self.statistics['calls']+=1;self.statistics['candidate_pairs']+=self.statistics['cached_pairs']
        return world if derivatives else world.gaps
