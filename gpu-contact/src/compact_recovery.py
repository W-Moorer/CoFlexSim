"""Surface-exact, spatially blocked volume recovery with a paired adjoint.

Only interior rows are compressed. Relative Frobenius truncation is an
operator-storage tolerance, not a trajectory accuracy certificate. Both the
forward displacement and force pullback use the SAME stored factors.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import scipy.linalg as la


class CompactRecovery:
    def __init__(self,shape,blocks,record):
        self.shape=tuple(shape);self.blocks=blocks;self.record=record
        self.nbytes=sum(rows.nbytes+A.nbytes+(0 if B is None else B.nbytes) for rows,A,B in blocks)
    def __matmul__(self,rhs):
        rhs=np.asarray(rhs);out=np.empty((self.shape[0],)+rhs.shape[1:])
        for rows,A,B in self.blocks:out[rows]=A@rhs if B is None else A@(B@rhs)
        return out
    @property
    def T(self):
        parent=self
        class Adjoint:
            def __matmul__(self,rhs):
                rhs=np.asarray(rhs);out=np.zeros((parent.shape[1],)+rhs.shape[1:])
                for rows,A,B in parent.blocks:
                    value=A.T@rhs[rows];out+=value if B is None else B.T@value
                return out
        return Adjoint()
    def dense(self):
        out=np.empty(self.shape)
        for rows,A,B in self.blocks:out[rows]=A if B is None else A@B
        return out
    def reshape(self,*shape):return self.dense().reshape(*shape)
    def __getitem__(self,key):return self.dense()[key]


def spatial_groups(points,indices,maximum):
    if len(indices)<=maximum:return [indices]
    axis=np.ptp(points[indices],axis=0).argmax();order=indices[np.argsort(points[indices,axis],kind='stable')]
    middle=len(order)//2
    return spatial_groups(points,order[:middle],maximum)+spatial_groups(points,order[middle:],maximum)


def build_recovery(matrix,positions,exact_rows,tolerance=1e-10,block_rows=768):
    if not 0<tolerance<1e-3:raise ValueError('Declare a small positive storage tolerance')
    matrix=np.asarray(matrix);n,m=matrix.shape;exact_rows=np.unique(np.asarray(exact_rows,int))
    remaining=np.setdiff1d(np.arange(n),exact_rows);groups=spatial_groups(positions,remaining,int(block_rows))
    blocks=[];discarded=0.;ranks=[]
    if len(exact_rows):blocks.append((exact_rows,matrix[exact_rows].copy(),None))
    for rows in groups:
        A=matrix[rows];U,s,V=la.svd(A,full_matrices=False,check_finite=False)
        tail=np.r_[np.cumsum((s*s)[::-1])[::-1],0.]
        rank=int(np.flatnonzero(tail<=tolerance**2*tail[0])[0]) if tail[0] else 0
        # A factorization must save at least 15% to justify another GEMV.
        if rank*(len(rows)+m)<.85*len(rows)*m:
            left=np.ascontiguousarray(U[:,:rank]*s[:rank]);right=np.ascontiguousarray(V[:rank])
            blocks.append((rows,left,right));discarded+=tail[rank];ranks.append(rank)
        else:blocks.append((rows,A.copy(),None));ranks.append(None)
    record=dict(storage='surface_exact_spatial_low_rank_v1',tolerance=tolerance,
        exact_row_count=len(exact_rows),dense_bytes=matrix.nbytes,interior_block_ranks=ranks,
        discarded_frobenius_norm=float(np.sqrt(discarded)),trajectory_accuracy_certified=False)
    op=CompactRecovery(matrix.shape,blocks,record);record['stored_bytes']=op.nbytes
    return op


def cached_recovery(matrix,positions,exact_rows,folder,tolerance=1e-10,block_rows=768):
    digest=hashlib.sha256(Path(__file__).read_bytes())
    for x in (matrix,positions,exact_rows):
        for part in np.array_split(x,max(1,(x.nbytes+4194303)//4194304)):
            digest.update(np.ascontiguousarray(part).tobytes())
    digest.update(str((tolerance,block_rows)).encode());key=digest.hexdigest()
    path=Path(folder)/('recovery_'+key+'.npz')
    if path.exists():
        with np.load(path,allow_pickle=False) as a:
            record=json.loads(str(a['record']));blocks=[]
            for i in range(record['block_count']):
                B=a['right_'+str(i)];blocks.append((a['rows_'+str(i)],a['left_'+str(i)],None if B.size==0 else B))
            return CompactRecovery(tuple(a['shape']),blocks,record)
    op=build_recovery(matrix,positions,exact_rows,tolerance,block_rows);op.record['block_count']=len(op.blocks)
    arrays=dict(shape=np.array(matrix.shape),record=json.dumps(op.record))
    for i,(rows,A,B) in enumerate(op.blocks):arrays.update({f'rows_{i}':rows,f'left_{i}':A,f'right_{i}':np.empty(0) if B is None else B})
    temporary=path.with_suffix('.pending.npz');np.savez_compressed(temporary,**arrays);temporary.replace(path)
    return op


class ModalRecovery:
    """Exact principal columns plus the compressed static recovery operator."""
    def __init__(self,principal,residual):
        self.principal=principal;self.residual=residual
        self.shape=(len(principal),principal.shape[1]+residual.shape[1])
        self.nbytes=principal.nbytes+residual.nbytes
    def __matmul__(self,rhs):
        r=self.principal.shape[1]
        return self.principal@rhs[:r]+self.residual@rhs[r:]
    @property
    def T(self):
        parent=self
        class Adjoint:
            def __matmul__(self,rhs):return np.concatenate((parent.principal.T@rhs,parent.residual.T@rhs),axis=0)
        return Adjoint()
    def dense(self):return np.c_[self.principal,self.residual.dense()]
    def reshape(self,*shape):return self.dense().reshape(*shape)
    def __getitem__(self,key):return self.dense()[key]


def runtime_archive(body,basis_source,policy,folder):
    """Identity for offline-finished modal operators; no dense volume reload."""
    from quasistatic_basis import DEFAULT_BASIS_POLICY
    settings={k:policy.get(k,v) for k,v in DEFAULT_BASIS_POLICY.items()}
    for k in ['residual_compression_rank','residual_compression_strategy','compact_recovery_tolerance','compact_recovery_block_rows']:
        settings[k]=policy.get(k)
    h=hashlib.sha256(json.dumps(settings,sort_keys=True).encode())
    for b in [body,basis_source]:
        for name,value in sorted(b.spec.items()):
            h.update(name.encode())
            if isinstance(value,np.ndarray):h.update(str((value.shape,value.dtype)).encode());h.update(value.tobytes())
            else:h.update(json.dumps(value,sort_keys=True).encode())
    for name in ['quasistatic_basis.py','residual_compression.py','compact_recovery.py','native.cpp','coflex_core.py','quasistatic_pd.py']:
        h.update(Path(__file__).with_name(name).read_bytes())
    return Path(folder)/('runtime_modal_'+h.hexdigest()+'.npz')


def save_runtime(path,body,recovery,reduced_L):
    arrays=dict(principal=recovery.principal,shape=np.array(recovery.residual.shape),
        record=json.dumps(recovery.residual.record),selection=json.dumps(body.selection),
        reduced_L=reduced_L,column_energy=body.reference_column_energy,
        surface_trace=body.surface_trace,material_cache=json.dumps(body.material_cache))
    for i,(rows,A,B) in enumerate(recovery.residual.blocks):
        arrays.update({f'rows_{i}':rows,f'left_{i}':A,f'right_{i}':np.empty(0) if B is None else B})
    temporary=path.with_suffix('.pending.npz');np.savez_compressed(temporary,**arrays);temporary.replace(path)


def load_runtime(path):
    if not path.exists():return None
    with np.load(path,allow_pickle=False) as a:
        record=json.loads(str(a['record']));blocks=[]
        for i in range(record['block_count']):
            B=a[f'right_{i}'];blocks.append((a[f'rows_{i}'],a[f'left_{i}'],None if not B.size else B))
        recovery=ModalRecovery(a['principal'],CompactRecovery(a['shape'],blocks,record))
        selection=json.loads(str(a['selection']));selection['runtime_archive_hit']=True
        return dict(recovery=recovery,selection=selection,reduced_L=a['reduced_L'],
            column_energy=a['column_energy'],surface_trace=a['surface_trace'],material_cache=json.loads(str(a['material_cache'])))
