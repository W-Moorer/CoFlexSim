"""Bounded conservative broadphase reuse; all narrowphase geometry stays fresh."""
import numpy as np
import time
from rtree import index


class CandidateRows:
    """CSR candidates, with a lazy sequence interface for legacy geometry."""
    def __init__(self,indices,offsets):self.indices=indices;self.offsets=offsets
    def __len__(self):return len(self.offsets)-1
    def __getitem__(self,i):
        if i<0:i+=len(self)
        if i<0 or i>=len(self):raise IndexError(i)
        return self.indices[self.offsets[i]:self.offsets[i+1]]
    @property
    def pair_count(self):return len(self.indices)


class ContactCandidateCache:
    def __init__(self,skin,max_bytes=16*1024**2,engine='rtree'):
        if skin<=0 or max_bytes<=0:raise ValueError('Positive cache skin and capacity required')
        if engine not in ('rtree','aabb_bvh'):raise ValueError('Unknown contact broadphase')
        self.skin=skin;self.max_bytes=max_bytes;self.entries={};self.engine=engine
        self.statistics={'rebuilds':0,'reuses':0,'capacity_fallbacks':0,'stored_bytes':0,'peak_stored_bytes':0}
        self.statistics.update(engine=engine,bvh_builds=0,bvh_refits=0,query_seconds=0.,index_update_seconds=0.,index_query_seconds=0.,filter_seconds=0.)

    def query(self,key,secondary,fs,main,fm,radius):
        tick=time.perf_counter()
        try:return self._query(key,secondary,fs,main,fm,radius)
        finally:self.statistics['query_seconds']+=time.perf_counter()-tick

    def _filter_timed(self,*args):
        tick=time.perf_counter()
        try:return self._filter(*args)
        finally:self.statistics['filter_seconds']+=time.perf_counter()-tick

    def _query(self,key,secondary,fs,main,fm,radius):
        signature=(fs.shape,fm.shape,hash(fs.tobytes()),hash(fm.tobytes()),radius)
        item=self.entries.get(key)
        sn=np.unique(fs);mn=np.unique(fm)
        if item is not None and item['signature']==signature:
            motion=float(np.abs(secondary[sn]-item['secondary']).max(initial=0.)+np.abs(main[mn]-item['main']).max(initial=0.))
            if motion<=self.skin*(1.-1e-12):
                self.statistics['reuses']+=1
                return self._filter_timed(item,secondary,fs,main,fm,radius)
        self.statistics['rebuilds']+=1
        if item is not None:self.statistics['stored_bytes']-=item['bytes'];del self.entries[key]
        tri=main[fm];low=tri.min(axis=1);high=tri.max(axis=1)
        offsets=[0];pairs=[];basebytes=secondary[sn].nbytes+main[mn].nbytes+8*(len(fs)+1)
        tree=None;tick=time.perf_counter()
        if self.engine=='aabb_bvh':
            import sys
            from coflex_core import BUILD
            if str(BUILD/'mortar_native/Release') not in sys.path:sys.path.insert(0,str(BUILD/'mortar_native/Release'))
            from coflex_mortar_native import AABBTree
            tree=item.get('tree') if item is not None and item['signature']==signature else None
            if tree is None:tree=AABBTree(low,high);self.statistics['bvh_builds']+=1
            else:tree.refit(low,high);self.statistics['bvh_refits']+=1
            basebytes+=tree.bytes
            self.statistics['index_update_seconds']+=time.perf_counter()-tick;tick=time.perf_counter()
            s=secondary[fs];budget=max(-1,(self.max_bytes-basebytes)//4)
            result=tree.query(s.min(axis=1)-radius-self.skin,s.max(axis=1)+radius+self.skin,budget) if budget>=0 else {'complete':False}
            self.statistics['index_query_seconds']+=time.perf_counter()-tick
            if not result['complete']:self.statistics['capacity_fallbacks']+=1;return None
            pairs=result['indices'];offsets=result['offsets']
        else:
            prop=index.Property();prop.dimension=3
            legacy=index.Index(((i,tuple(np.concatenate((a,b))),None) for i,(a,b) in enumerate(zip(low,high))),properties=prop)
            self.statistics['index_update_seconds']+=time.perf_counter()-tick;tick=time.perf_counter()
            try:
                for face in fs:
                    s=secondary[face];box=np.concatenate((s.min(axis=0)-radius-self.skin,s.max(axis=0)+radius+self.skin))
                    pairs.extend(sorted(legacy.intersection(tuple(box))));offsets.append(len(pairs))
                    if basebytes+4*len(pairs)>self.max_bytes:
                        self.statistics['capacity_fallbacks']+=1;return None
            finally:legacy.close();self.statistics['index_query_seconds']+=time.perf_counter()-tick
        item={'signature':signature,'secondary':secondary[sn].copy(),'main':main[mn].copy(),
              'indices':np.asarray(pairs,dtype=np.int32),'offsets':np.asarray(offsets,dtype=np.int64),'tree':tree}
        item['bytes']=basebytes+item['indices'].nbytes
        while self.entries and self.statistics['stored_bytes']+item['bytes']>self.max_bytes:
            victim=next(iter(self.entries));self.statistics['stored_bytes']-=self.entries.pop(victim)['bytes']
        self.entries[key]=item;self.statistics['stored_bytes']+=item['bytes']
        self.statistics['peak_stored_bytes']=max(self.statistics['peak_stored_bytes'],self.statistics['stored_bytes'])
        return self._filter_timed(item,secondary,fs,main,fm,radius)

    @staticmethod
    def _filter(item,secondary,fs,main,fm,radius):
        tri=main[fm];low=tri.min(axis=1);high=tri.max(axis=1)
        secondary_tri=secondary[fs];lo=secondary_tri.min(axis=1)-radius;hi=secondary_tri.max(axis=1)+radius
        ids=item['indices'];owners=np.repeat(np.arange(len(fs)),np.diff(item['offsets']))
        valid=np.all(high[ids]>=lo[owners],axis=1)&np.all(low[ids]<=hi[owners],axis=1)
        # Keep original face/candidate order, including empty rows. The vector
        # predicates are exactly the previous per-face AABB tests.
        prefix=np.r_[0,np.cumsum(valid)]
        return CandidateRows(ids[valid],prefix[item['offsets']])
