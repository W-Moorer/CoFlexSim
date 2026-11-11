"""GPU BVH, finite-overlap P1 pressure rows and physical vertex guards.

Same discretization as SharedBoundedContact. Current surface coordinates, all
row data, integration and derivative assembly stay on CUDA. Host only dispatches.
"""
import cupy as cp
import cupyx.scipy.sparse as csp
import numpy as np
from device_bvh import DeviceBVH
from device_overlap import DeviceOverlap
from fast_assembly import FastAssembly
from geometry_graph import PairGeometryGraph


class DeviceWorld:
    def __init__(self,gaps,G,keys,physical_min,stats):
        self.gaps=gaps;self.G=G;self.keys=keys;self.physical_min=physical_min;self.stats=stats


class DeviceGeometry:
    def __init__(self,bodies,scene,use_graph=False):
        self.bodies=bodies;self.offsets=np.r_[0,np.cumsum([len(b.X) for b in bodies])]
        self.faces=[cp.asarray(b.spec['contact_faces'],dtype=cp.int32) for b in bodies]
        self.trees=[DeviceBVH(b.X,b.spec['contact_faces']) for b in bodies]
        self.radius=float(scene.get('contact_search_distance',.001));self.bound=scene['contact_patch_bound_m']
        self.guard_offset=self.bound-scene.get('contact_guard_reserve_m',0.)
        self.asymmetric=scene.get('contact_pressure_support','legacy_symmetric')=='upper_bounded'
        self.band=scene.get('contact_pressure_band_m',(.5 if self.asymmetric else 2)*self.bound)
        self.pairs=sorted(scene['contact_pairs']);self.kernel=DeviceOverlap();self.assembler=FastAssembly()
        self.size=int(self.offsets[-1]*3);self.maxnodes=int(self.offsets[-1])+1
        self.maxfaces=max(map(len,self.faces))+1;self.guard_base=len(bodies)**2*self.maxnodes
        self.statistics=dict(calls=0,candidate_pairs=0,overlap_pairs=0,online_geometry='cuda_finite_overlap_P1')
        self.history_keys=cp.empty(0,dtype=cp.int64);self.history_values=cp.empty(0)
        self.graph_pairs={tuple(pair):PairGeometryGraph(self,*pair) for pair in self.pairs} if use_graph else {}

    def build(self,x,derivatives=True):
        wg=[];gg=[];wG=[];gG=[];wk=[];gk=[];minima=[]
        if self.graph_pairs:
            for pair in self.pairs:
                packed,mat,keys,physical,nw,nc,no=self.graph_pairs[tuple(pair)].build(x,derivatives)
                self.statistics['candidate_pairs']+=nc;self.statistics['overlap_pairs']+=no
                if len(self.pairs)==1:
                    self.statistics['calls']+=1
                    return DeviceWorld(packed,mat,keys,physical,self.statistics) if derivatives else packed
                wg.append(packed[:nw]);gg.append(packed[nw:]);minima.append(physical)
                if derivatives:wG.append(mat[:nw]);gG.append(mat[nw:]);wk.append(keys[:nw]);gk.append(keys[nw:])
            gaps=cp.concatenate(wg+gg) if wg else cp.empty(0);self.statistics['calls']+=1
            if not derivatives:return gaps
            G=csp.vstack(wG+gG,format='csr') if wG else csp.csr_matrix((0,self.size))
            keys=cp.concatenate(wk+gk) if wk else cp.empty(0,dtype=cp.int64)
            return DeviceWorld(gaps,G,keys,cp.min(cp.stack(minima)) if minima else cp.asarray(0.),self.statistics)
        for b in sorted(set(p[1] for p in self.pairs)):
            self.trees[b].refit(x[self.offsets[b]:self.offsets[b+1]])
        for slave,master in self.pairs:
            fs,fm=self.faces[slave],self.faces[master]
            sm=x[self.offsets[slave]:self.offsets[slave+1]];mm=x[self.offsets[master]:self.offsets[master+1]]
            owners,mids=self.trees[master].query(sm,fs,self.radius)
            self.statistics['candidate_pairs']+=len(owners)
            S=sm[fs[owners]];M=mm[fm[mids]]
            data=self.kernel.build(S,M,self.radius,self.band,self.asymmetric,derivatives)
            selected=data['selected'];owners=owners[selected];mids=mids[selected]
            self.statistics['overlap_pairs']+=len(selected)
            if not len(selected):continue
            nodes=cp.concatenate((fs[owners]+int(self.offsets[slave]),fm[mids]+int(self.offsets[master])),axis=1)
            paircode=slave*len(self.bodies)+master
            packed,mat,keys,physical,nw=self.assembler.build(data,owners,mids,nodes,x,len(fs),self.size,self.maxnodes,
                paircode,self.maxfaces,self.guard_base,self.guard_offset,self.radius,derivatives)
            if len(self.pairs)==1:
                self.statistics['calls']+=1
                return DeviceWorld(packed,mat,keys,physical,self.statistics) if derivatives else packed
            wg.append(packed[:nw]);gg.append(packed[nw:]);minima.append(physical)
            if derivatives:
                # Maintain the original order: all pressure rows, then guards.
                wG.append(mat[:nw]);gG.append(mat[nw:]);wk.append(keys[:nw]);gk.append(keys[nw:])
        gaps=cp.concatenate(wg+gg) if wg else cp.empty(0)
        self.statistics['calls']+=1
        if not derivatives:return gaps
        G=csp.vstack(wG+gG,format='csr') if wG else csp.csr_matrix((0,self.size))
        keys=cp.concatenate(wk+gk) if wk else cp.empty(0,dtype=cp.int64)
        minimum=cp.min(cp.stack(minima)) if minima else cp.asarray(0.)
        return DeviceWorld(gaps,G,keys,minimum,self.statistics)

    def seeds(self,world,free_gap):
        if not len(self.history_keys):return {}
        pos=cp.searchsorted(self.history_keys,world.keys);safe=cp.minimum(pos,len(self.history_keys)-1)
        valid=(world.keys>=0)&(pos<len(self.history_keys))&(self.history_keys[safe]==world.keys)&(free_gap<0)
        ids=cp.flatnonzero(valid)
        return ids,self.history_values[safe[ids]]

    def commit(self,world,lam):
        ids=cp.flatnonzero((lam>1e-8)&(world.keys>=0));order=cp.argsort(world.keys[ids])
        self.history_keys=world.keys[ids][order];self.history_values=lam[ids][order]
