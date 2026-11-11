"""Offline conservative far-row bounds of the actual corrected surface operator.

Streams one triangular panel at a time; stores only O(surface DOFs) bounds.
Reuses the original bank and retained-space factor, with no FE solve or new bank.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp


def signature(source,faces):
    h=hashlib.sha256()
    for a in (source.U,np.asarray(faces,dtype=np.int64)):
        h.update(memoryview(np.ascontiguousarray(a)).cast('B'))
    h.update(json.dumps(source.source.catalog_state.get('source_identity',{}),sort_keys=True).encode())
    return h.hexdigest()


def prepare(source,faces,path=None,panel=384,progress=None):
    import cupy as cp
    tick=time.perf_counter();faces=np.asarray(faces,dtype=int);sha=signature(source,faces)
    if path is not None:
        path=Path(path)
        if path.exists():
            with np.load(path,allow_pickle=False) as a:
                if str(a['signature'])!=sha:raise ValueError('Certificate identity changed; do not overwrite')
                source.far_column_bound=a['bound'].copy()
            return dict(loaded=True,wall_s=time.perf_counter()-tick,path=str(path),signature=sha)
    n=source.n;k=source.U.shape[1]
    edges=np.asarray([(i,j) for f in faces for i in f for j in f],dtype=int)
    near=sp.coo_matrix((np.ones(len(edges),dtype=bool),(edges[:,0],edges[:,1])),shape=(n//3,n//3)).tocsr()
    U=cp.asarray(source.U);norm=np.linalg.norm(source.U,axis=1);bound=np.zeros(n)
    eps=np.finfo(np.float64).eps;gamma=(k+3)*eps/(1-(k+3)*eps)
    previous=source.far_column_bound.copy()
    for first in range(0,n,panel):
        end=min(first+panel,n);rows=np.arange(end);cols=np.arange(first,end)
        values=cp.asarray(source.source.block(rows,cols))+U[:end]@U[first:end].T
        mask=near[rows//3][:,cols//3].toarray()
        values=cp.where(cp.asarray(mask),0.,cp.abs(values))
        # Outward allowance for the FP64 dot product and addition, including
        # both triangle directions; it is not fitted to a trajectory.
        vmax_col=cp.asnumpy(cp.max(values,axis=0));vmax_row=cp.asnumpy(cp.max(values,axis=1))
        vmax_col+=gamma*norm[:end].max(initial=0.)*norm[first:end]+eps*vmax_col
        vmax_row+=gamma*norm[first:end].max(initial=0.)*norm[:end]+eps*vmax_row
        bound[first:end]=np.maximum(bound[first:end],vmax_col);bound[:end]=np.maximum(bound[:end],vmax_row)
        if progress is not None:progress(end,n)
    bound=np.nextafter(bound,np.inf);source.far_column_bound=np.minimum(previous,bound)
    cp.cuda.get_current_stream().synchronize();del U,values;cp.get_default_memory_pool().free_all_blocks()
    if path is not None:
        path.parent.mkdir(exist_ok=True,parents=True)
        np.savez_compressed(path,bound=source.far_column_bound,signature=np.asarray(sha))
    return dict(loaded=False,wall_s=time.perf_counter()-tick,signature=sha,
        maximum_bound_ratio=float(source.far_column_bound.max()/previous.max()),
        median_bound_ratio=float(np.median(source.far_column_bound/previous)),
        whole_operator_scanned=True,new_dense_library=False,path=str(path) if path else None)
