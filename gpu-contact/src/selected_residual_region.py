"""Read-only global residual bank plus a bounded retained-mode correction.

No eigenvalue extraction, FE factorization or second surface library. For a
subset of retained coordinates the omitted static response is a PSD Schur
complement update U U^T. Its frequently queried region is prepared once on the
host; misses remain exact and are charged online. CPU and CUDA use block().
"""
import time
import numpy as np
import scipy.linalg as la
from threadpoolctl import threadpool_limits


class SelectedResidualRegion:
    sealed=True
    def __init__(self,source,body,keep,threads=4):
        tick=time.perf_counter();self.source=source;self.n=source.n;self.threads=threads
        keep=np.asarray(keep,dtype=int);drop=np.setdiff1d(np.arange(body.P.shape[1]),keep)
        if len(np.unique(keep))!=len(keep) or np.any(keep<0) or np.any(keep>=body.P.shape[1]):raise ValueError('Invalid mode subset')
        with threadpool_limits(threads):
            if len(drop):
                cross=la.solve(body.K[np.ix_(keep,keep)],body.K[np.ix_(keep,drop)],assume_a='pos')
                schur=body.K[np.ix_(drop,drop)]-body.K[np.ix_(drop,keep)]@cross
                schur=(schur+schur.T)/2
                L=la.cholesky(schur,lower=True)
                q=body.P[:,drop]-body.P[:,keep]@cross
                self.U=np.ascontiguousarray(la.solve_triangular(L,q.T,lower=True).T)
            else:self.U=np.zeros((self.n,0))
        norm=np.linalg.norm(self.U,axis=1)
        # Cauchy-Schwarz gives an all-row bound and hence a valid far-row bound.
        self.far_column_bound=np.nextafter(np.asarray(source.far_column_bound)+norm.max(initial=0.)*norm,np.inf)
        self.ids=np.empty(0,dtype=int);self.lookup=np.full(self.n,-1,dtype=int);self.correction=np.empty((0,0))
        self.statistics=dict(source='one_global_bank_plus_retained_schur_region',stored_bytes=source.statistics['stored_bytes'],
            factor_bytes=self.U.nbytes,correction_cache_bytes=0,offline_s=time.perf_counter()-tick,
            online_factor_solves=0,block_requests=0,query_s=0.,uncached_update_s=0.,uncached_entries=0,
            global_bank_unchanged=True)

    def prewarm(self,priority_ids,budget_bytes=256*2**20):
        """Initialization only; priority comes from a separate original-mode pilot."""
        tick=time.perf_counter();priority_ids=np.asarray(priority_ids,dtype=int)
        if np.any(priority_ids<0) or np.any(priority_ids>=self.n):raise IndexError('Region outside bank')
        _,first=np.unique(priority_ids,return_index=True);priority_ids=priority_ids[np.sort(first)]
        capacity=int(np.sqrt(budget_bytes//8));self.ids=priority_ids[:capacity].copy()
        self.lookup.fill(-1);self.lookup[self.ids]=np.arange(len(self.ids))
        with threadpool_limits(self.threads):
            u=self.U[self.ids];self.correction=u@u.T
        self.statistics.update(correction_cache_bytes=self.correction.nbytes,prepared_region_dofs=len(self.ids),
            requested_region_dofs=len(priority_ids),region_precompute_s=time.perf_counter()-tick)
        self.statistics['offline_s']+=time.perf_counter()-tick
        return self.statistics.copy()

    def block(self,rows,cols):
        tick=time.perf_counter();rows=np.asarray(rows,dtype=int).ravel();cols=np.asarray(cols,dtype=int).ravel()
        out=self.source.block(rows,cols);ri=self.lookup[rows];ci=self.lookup[cols]
        good_r=np.flatnonzero(ri>=0);bad_r=np.flatnonzero(ri<0);good_c=np.flatnonzero(ci>=0);bad_c=np.flatnonzero(ci<0)
        if len(good_r) and len(good_c):out[np.ix_(good_r,good_c)]+=self.correction[np.ix_(ri[good_r],ci[good_c])]
        if len(bad_r) or (len(good_r) and len(bad_c)):
            update=time.perf_counter()
            with threadpool_limits(self.threads):
                if len(bad_r):out[bad_r]+=self.U[rows[bad_r]]@self.U[cols].T
                if len(good_r) and len(bad_c):out[np.ix_(good_r,bad_c)]+=self.U[rows[good_r]]@self.U[cols[bad_c]].T
            self.statistics['uncached_entries']+=len(bad_r)*len(cols)+len(good_r)*len(bad_c)
            self.statistics['uncached_update_s']+=time.perf_counter()-update
        self.statistics['block_requests']+=1;self.statistics['query_s']+=time.perf_counter()-tick
        return out

    def apply(self,force,rows=None):
        rows=np.arange(self.n) if rows is None else rows;ids=np.flatnonzero(force)
        return self.block(rows,ids)@force[ids]

    def columns(self,ids):
        """Exact CPU reference interface, using the same selected-space bank."""
        return self.block(np.arange(self.n),ids)

    def reset_online_statistics(self):
        for key in ('block_requests','query_s','uncached_update_s','uncached_entries'):self.statistics[key]=0
