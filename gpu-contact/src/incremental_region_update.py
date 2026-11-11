"""Exact incremental R[I,I] updates with resident retained-space factors.

No scene dispatch. CPU uses the source's exact block interface; CUDA consumes
the same R_base + U U^T decomposition. Existing region entries survive growth.
Only newly needed cross-region columns are formed, once, on the GPU.
"""
import time
import numpy as np
import cupy as cp
from device_simulator import sync
from response_cache_policy import plan_slots


def rank(source):
    return source.U.shape[1] if hasattr(source,'U') else 0


def bytes_for(body,source,capacity):
    return 8*capacity*(capacity+body.P.shape[1]+rank(source))


def update(sim,requested=None):
    sync();start=time.perf_counter()
    if requested is None:requested=sim.candidate_columns()
    if not hasattr(sim,'incremental_statistics'):
        sim.incremental_statistics=dict(calls=0,resizes=0,preserved_columns=0,new_columns=0,evicted_columns=0,
            host_Schur_expansions=0,correction_upload_bytes=0,gpu_correction_s=0.,base_query_s=0.)
    stats=sim.incremental_statistics
    sizes=lambda caps:sum(bytes_for(b,s,c) for b,s,c in zip(sim.bodies,sim.sources,caps))
    minima=[max(1,len(ids)) for ids in requested]
    if sizes(minima)>sim.window_budget:raise MemoryError('Exact regional blocks and factors exceed declared budget')
    capacities=[B.C.shape[0] if B.C.shape[0]>=m else min(len(b.P),max(1,64*((3*m//2+63)//64)))
        for b,B,m in zip(sim.bodies,sim.dynamic,minima)]
    if sizes(capacities)>sim.window_budget:capacities=minima
    if sim.scene.get('regional_reserve_arenas',False):
        # Fixed bounded arenas are planned from model sizes and the declared
        # memory budget, not from a future trajectory or observed contact force.
        low,high=1,max(len(b.P) for b in sim.bodies)
        while low<high:
            mid=(low+high+1)//2
            trial=[min(len(b.P),max(m,mid)) for b,m in zip(sim.bodies,minima)]
            if sizes(trial)<=sim.window_budget:low=mid
            else:high=mid-1
        capacities=[min(len(b.P),max(m,low)) for b,m in zip(sim.bodies,minima)]
    if sim.scene.get('regional_fill_budget',False) and any(B.C.shape[0]<m for B,m in zip(sim.dynamic,minima)):
        # Spend the same bounded regional budget on reusable history/prefetch.
        # This does not enlarge the exact operator or select fewer constraints.
        low,high=1.,4.
        for _ in range(32):
            ratio=(low+high)*.5
            trial=[min(len(b.P),max(m,int(ratio*m))) for b,m in zip(sim.bodies,minima)]
            if sizes(trial)<=sim.window_budget:low=ratio
            else:high=ratio
        capacities=[min(len(b.P),max(m,int(low*m))) for b,m in zip(sim.bodies,minima)]
    changed=False;counts=[]
    for b,B,source,ids,capacity in zip(sim.bodies,sim.dynamic,sim.sources,requested,capacities):
        ids=np.unique(ids);k=rank(source)
        if B.C.shape!=(capacity,capacity) or not hasattr(B,'regional_schur_factor'):
            oldC=B.C;oldD=B.regional_modal_response;oldU=getattr(B,'regional_schur_factor',None)
            slots=B.window_slots;valid=np.flatnonzero(slots>=0)
            # Preserve active entries first, then spare history. This also
            # handles a tighter shared budget without losing active data.
            active=valid[np.isin(slots[valid],ids)];history=valid[~np.isin(slots[valid],ids)]
            take=np.r_[active,history][:capacity];kept=slots[take]
            sim.allocate_region(B,capacity)
            B.regional_schur_factor=cp.zeros((capacity,k))
            if len(take):
                oldix=cp.asarray(take);n=len(take)
                B.C[:n,:n]=oldC[cp.ix_(oldix,oldix)];B.regional_modal_response[:n]=oldD[oldix]
                if k and oldU is not None:B.regional_schur_factor[:n]=oldU[oldix]
            B.window_slots=np.full(capacity,-1,dtype=np.int64);B.window_slots[:len(kept)]=kept
            B.window_lookup.fill(-1)
            if len(kept):B.window_lookup[cp.asarray(kept)]=cp.arange(len(kept),dtype=cp.int32)
            stats['resizes']+=1;stats['preserved_columns']+=len(kept);changed=True
            del oldC,oldD,oldU
        missing,destinations,evicted=plan_slots(B.window_slots,ids)
        if sim.scene.get('regional_retain_spare',False) and hasattr(B,'history_age'):
            empty=np.flatnonzero(B.window_slots<0)
            inactive=np.flatnonzero((B.window_slots>=0)&~np.isin(B.window_slots,ids))
            inactive=inactive[np.argsort(B.history_age[B.window_slots[inactive]],kind='stable')]
            destinations=np.r_[empty,inactive][:len(missing)]
            evicted=B.window_slots[destinations];evicted=evicted[evicted>=0]
        stats['evicted_columns']+=len(evicted)
        if len(missing):
            if len(evicted):B.window_lookup[cp.asarray(evicted)]=-1
            new_slots=B.window_slots.copy();new_slots[destinations]=missing
            occupied=np.flatnonzero(new_slots>=0);all_ids=new_slots[occupied]
            base=source.source if hasattr(source,'U') else source
            query=base.packed_query if getattr(sim,'compiled_region_query',False) else base
            tick=time.perf_counter()
            bounded=sim.scene.get('regional_bounded_refill',False)
            if bounded:
                from bounded_refill import BoundedRefill
                if not hasattr(sim,'refill_staging'):
                    sim.refill_staging=BoundedRefill(sim.scene.get('regional_refill_slot_bytes',4*2**20))
                sim.refill_staging.update(B,source,query,all_ids,missing,occupied,destinations)
                sync()
                stats.setdefault('bounded_refill_s',0.)
                stats['bounded_refill_s']+=time.perf_counter()-tick
                stats['correction_upload_bytes']+=len(missing)*k*8 if not hasattr(source,'device_factor') else 0
            elif hasattr(query,'block_gpu'):
                values=query.block_gpu(all_ids,missing);sync()
            else:values=cp.asarray(query.block(all_ids,missing))
            if not bounded:stats['base_query_s']+=time.perf_counter()-tick
            ii,jj=cp.asarray(occupied),cp.asarray(destinations)
            if k and not bounded:
                sync();tick=time.perf_counter()
                if hasattr(source,'device_factor'):
                    B.regional_schur_factor[jj]=source.device_factor[cp.asarray(missing)]
                    uploaded_bytes=0
                else:
                    factor=source.U[missing];B.regional_schur_factor[jj]=cp.asarray(factor);uploaded_bytes=factor.nbytes
                values+=B.regional_schur_factor[ii]@B.regional_schur_factor[jj].T
                sync();stats['gpu_correction_s']+=time.perf_counter()-tick
                stats['correction_upload_bytes']+=uploaded_bytes
            if not bounded:
                B.C[cp.ix_(ii,jj)]=values;B.C[cp.ix_(jj,ii)]=values.T
            B.regional_modal_response[jj]=B.P[cp.asarray(missing)]@B.inverse.T
            B.window_slots=new_slots;B.window_lookup[cp.asarray(missing)]=cp.asarray(destinations,dtype=cp.int32)
            stats['new_columns']+=len(missing)
        B.compact_ids=ids.copy();B.window_ids=B.window_slots[B.window_slots>=0]
        counts.append(dict(surface_dofs=len(b.P),candidate_columns=len(ids),new_columns=len(missing),
            cached_columns=len(B.window_ids),capacity=capacity,matrix_bytes=B.C.nbytes,
            regional_modal_factor_bytes=B.regional_modal_response.nbytes,
            regional_schur_factor_bytes=B.regional_schur_factor.nbytes,online_factor_solves=0))
    sync();used=sizes(capacities);sim.window_highwater=max(sim.window_highwater,used);stats['calls']+=1
    sim.window_events.append(dict(time_s=sim.t,wall_s=time.perf_counter()-start,operators=counts,
        bytes=used,sealed_device_operator_bytes=0,reallocated=changed,incremental_update=True))
    sim.window_ready=True
    if changed and hasattr(sim,'resident'):sim.resident.signature=None
    if changed and hasattr(sim,'graph_working'):sim.graph_working.graph=None
    return changed
