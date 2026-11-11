"""Fused GPU implementation of the unchanged smoothed median normal contact."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import cupy as cp
from device_bvh import DeviceBVH
from device_geometry import DeviceGeometry,DeviceWorld
from consistent_mortar_contact import ContactGeometryError

class FusedNormalContact:
    seeds=DeviceGeometry.seeds
    commit=DeviceGeometry.commit
    def __init__(self,bodies,scene):
        self.bodies=bodies;self.offsets=np.r_[0,np.cumsum([len(b.X) for b in bodies])]
        self.faces=[cp.asarray(b.spec['contact_faces'],dtype=cp.int32) for b in bodies]
        self.trees=[DeviceBVH(b.X,b.spec['contact_faces']) for b in bodies]
        self.radius=float(scene.get('contact_search_distance',.001));self.band=np.nextafter(self.radius,0.)
        self.pairs=sorted(scene['contact_pairs']);self.size=int(self.offsets[-1]*3);self.maxnodes=int(self.offsets[-1])+1
        module=cp.RawModule(code=Path(__file__).with_name('overlap.cu').read_text()+Path(__file__).with_name('normal_contact.cu').read_text(),options=('--std=c++11',))
        self.k={n:module.get_function(n) for n in ('nodal_normals','normal_patches','check_normal_coverage','pack_normal_rows')}
        self.history_keys=cp.empty(0,dtype=cp.int64);self.history_values=cp.empty(0)
        self.statistics=dict(calls=0,candidate_pairs=0,online_geometry='cuda_fused_smoothed_normal_median',
            force_kind='integrated_normal_traction',pointwise_guard_enforced=False)
    @staticmethod
    def launch(kernel,n,args):
        if n:kernel(((n+127)//128,),(128,),args)
    def build(self,x,derivatives=True):
        normals={}
        for slave in sorted(set(p[0] for p in self.pairs)):
            sm=x[self.offsets[slave]:self.offsets[slave+1]];fs=self.faces[slave];normals[slave]=cp.zeros_like(sm)
            self.launch(self.k['nodal_normals'],len(fs),(sm,fs,normals[slave],np.int32(len(fs))))
        for master in sorted(set(p[1] for p in self.pairs)):
            self.trees[master].refit(x[self.offsets[master]:self.offsets[master+1]])
        parts=[];minimum=cp.asarray(0.);error=cp.zeros(1,dtype=cp.int32)
        for slave,master in self.pairs:
            fs,fm=self.faces[slave],self.faces[master]
            sm=x[self.offsets[slave]:self.offsets[slave+1]];mm=x[self.offsets[master]:self.offsets[master+1]]
            owners,mids=self.trees[master].query(sm,fs,self.radius);n=len(owners);npiece=6*n
            self.statistics['candidate_pairs']+=n
            area=cp.zeros(len(sm));integral=cp.zeros(len(sm));counts=cp.zeros(len(sm),dtype=cp.int32)
            covered=cp.zeros(12*len(fs));reference=cp.zeros(6*len(fs))
            piece_values=cp.empty((npiece,18));piece_nodes=cp.empty((npiece,6),dtype=cp.int32);piece_row=cp.empty(npiece,dtype=cp.int32)
            self.launch(self.k['normal_patches'],npiece,(sm,mm,fs,fm,owners,mids,normals[slave],np.int32(n),np.float64(self.radius),np.float64(self.band),
                area,integral,counts,covered,reference,np.int32(len(fs)),piece_values,piece_nodes,piece_row,minimum,error))
            self.launch(self.k['check_normal_coverage'],6*len(fs),(covered,reference,np.int32(len(fs)),error))
            keep=cp.flatnonzero(area>1e-24);gaps=integral[keep]/area[keep]
            keys=((slave*len(self.bodies)+master)*self.maxnodes+keep).astype(cp.int64)
            if derivatives:
                ptr=cp.concatenate((cp.zeros(1,dtype=cp.int32),cp.cumsum(counts[keep]*18,dtype=cp.int32)))
                # Capacity beyond indptr[-1] is inert, preserving xyz triplets.
                cols=cp.zeros(npiece*18,dtype=cp.int32);values=cp.zeros(npiece*18)
                lut=cp.full(len(sm),-1,dtype=cp.int32);lut[keep]=cp.arange(len(keep),dtype=cp.int32);cursor=cp.zeros(len(keep),dtype=cp.int32)
                self.launch(self.k['pack_normal_rows'],npiece,(piece_row,piece_nodes,piece_values,area,lut,ptr,cursor,cols,values,
                    np.int32(npiece),np.int32(self.offsets[slave]),np.int32(self.offsets[master])))
                mat=SimpleNamespace(indptr=ptr,indices=cols,data=values,shape=(len(keep),self.size))
            else:mat=None
            parts.append((gaps,keys,mat))
        code=int(error[0])
        if code:raise ContactGeometryError('Fused normal geometry validity flags '+str(code))
        self.statistics['calls']+=1
        if len(parts)==1:
            gaps,keys,mat=parts[0]
        else:
            gaps=cp.concatenate([p[0] for p in parts]) if parts else cp.empty(0)
            keys=cp.concatenate([p[1] for p in parts]) if parts else cp.empty(0,dtype=cp.int64)
            if derivatives:
                # Truncate each capacity tail when concatenating multiple pairs.
                values=[];cols=[];pointers=[cp.zeros(1,dtype=cp.int32)];offset=0
                for _,_,m in parts:
                    used=int(m.indptr[-1]);values.append(m.data[:used]);cols.append(m.indices[:used]);pointers.append(m.indptr[1:]+offset);offset+=used
                mat=SimpleNamespace(indptr=cp.concatenate(pointers),indices=cp.concatenate(cols) if cols else cp.empty(0,dtype=cp.int32),
                    data=cp.concatenate(values) if values else cp.empty(0),shape=(len(gaps),self.size))
        return DeviceWorld(gaps,mat,keys,minimum,self.statistics) if derivatives else gaps
