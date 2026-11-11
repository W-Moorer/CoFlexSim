"""Optional compliance-spectrum compression, audited independently of trajectories.

Physical residual columns are K-orthonormalized first. A displacement-optimal
subspace is extracted by randomized SVD; its compliance is a PSD suboperator
of the original static compliance. This is an APPROXIMATION, not exact reuse.
The old 5% energy-response certificate is never inherited by this archive.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from quasistatic_basis import make_channels,DEFAULT_BASIS_POLICY,energy_error,elastic_factor


def body_signature(body):
    h=hashlib.sha256(b'compressed-residual-body-v1')
    for name in sorted(body.spec):
        v=body.spec[name];h.update(name.encode())
        if isinstance(v,np.ndarray):h.update(str((v.shape,v.dtype)).encode());h.update(v.tobytes())
        else:h.update(json.dumps(v,sort_keys=True).encode())
    for A in [body.K,body.M]:
        for v in [A.data,A.indices,A.indptr]:h.update(v.tobytes())
    return h.hexdigest()


def surface_tractions(body):
    """Consistent P1 normal traction patches on the complete potential surface.

    No engagement snapshot or active force is used. These are training loads,
    not a change to the online gap derivative or contact force.
    """
    faces=body.spec['contact_faces'];nodes=np.unique(faces);rows=[];cols=[];data=[];areas=np.zeros(len(nodes))
    lookup={int(n):i for i,n in enumerate(nodes)}
    for face in faces:
        cross=np.cross(body.X[face[1]]-body.X[face[0]],body.X[face[2]]-body.X[face[0]])
        area=la.norm(cross)/2;normal=cross/(2*area)
        for i,node in enumerate(face):
            col=lookup[int(node)];areas[col]+=area/3
            for j,target in enumerate(face):
                rows.extend((3*target+np.arange(3)).tolist());cols.extend([col]*3)
                data.extend((area/12*(2 if i==j else 1)*normal).tolist())
    F=sp.csr_matrix((data,(rows,cols)),shape=(body.X.size,len(nodes)))
    return F[body.free]@sp.diags(1/np.maximum(areas,1e-30))


def compress_residual(body,P,Z,rank,seed=20261102,strategy='displacement'):
    """Return K-orthogonal residual Ritz columns (no new physical force model)."""
    rank=min(int(rank),Z.shape[1]);rng=np.random.default_rng(seed)
    if rank==Z.shape[1]:
        return Z.copy(),dict(rank=rank,original_rank=rank,construction='unchanged_parent_space_within_rank_cap',PSD_suboperator_of_parent_compliance=True,old_response_certificate_inherited=True)
    if strategy=='traction' and not len(body.spec.get('contact_faces',[])):strategy='displacement'
    energy=np.sum(Z*(body.K@Z),axis=0)
    # Original construction is K-orthogonal but M-scaled. Verify the input
    # through independent random products without making another huge Gram.
    W=Z/np.sqrt(energy)[None,:]
    probes=rng.normal(size=(Z.shape[1],8))
    defect=la.norm(W.T@(body.K@(W@probes))-probes)/la.norm(probes)
    if defect>1e-7:raise ValueError('Input residual not K-orthonormal')
    width=min(rank+24,W.shape[1])
    if strategy=='displacement':
        Q=la.qr(W@rng.normal(size=(W.shape[1],width)),mode='economic')[0]
        Q=la.qr(W@(W.T@Q),mode='economic')[0]
        _,values,Vh=la.svd(Q.T@W,full_matrices=False,check_finite=False)
        V=Vh[:rank].T
    elif strategy=='traction':
        F=surface_tractions(body);B=(F.T@W).T
        # POD in physical elastic energy, weighted by all consistent normal
        # traction channels. Keep arbitrary free PD right-hand sides online.
        Q=la.qr(B@rng.normal(size=(B.shape[1],min(width,B.shape[1]))),mode='economic')[0]
        Q=la.qr(B@(B.T@Q),mode='economic')[0]
        U,values,_=la.svd(Q.T@B,full_matrices=False,check_finite=False)
        V=Q@U[:,:rank]
    else:raise ValueError(strategy)
    out=W@V
    mass=np.sqrt(np.sum(out*(body.M@out),axis=0));out/=mass[None,:]
    return out,dict(rank=rank,original_rank=Z.shape[1],seed=seed,sketch_width=width,
        input_K_orthogonality_probe_error=float(defect),retained_compliance_singular_values=values[:rank].tolist(),
        construction='global_physical_residual_'+strategy+'_spectrum_no_trajectory_snapshots',
        PSD_suboperator_of_parent_compliance=True,old_response_certificate_inherited=False)


def audit_compression(body,P,Z):
    policy=DEFAULT_BASIS_POLICY|dict(validation_loads=18)
    loads=make_channels(body,policy,20261103)
    rigid=P[:,np.sum(P*(body.K@P),axis=0)<1e-6]
    exact=elastic_factor(body.K,body.M,rigid)(loads)
    T=np.c_[P,Z];small=T.T@(body.K@T)
    # Rigid components are absent for supported gear/compression fixtures.
    approx=T@la.lstsq(small,T.T@loads,cond=1e-13)[0]
    return dict(untouched_load_seed=20261103,load_count=loads.shape[1],
        static_energy_error=energy_error(approx-exact,exact,body.K),
        maximum_displacement_relative_L2=float(np.max(la.norm(approx-exact,axis=0)/np.maximum(la.norm(exact,axis=0),1e-30))),
        acceptance='diagnostic_only_trajectory_and_response_accuracy_must_be_reverified')


def load_archive(body,folder):
    import io
    if isinstance(folder,dict):
        signature=body_signature(body);raw=folder[signature]
        with np.load(io.BytesIO(raw),allow_pickle=False) as data:
            P=data['Phi'];Z=data['Z'];record=json.loads(str(data['record']))
        if record['body_signature']!=signature or not np.isfinite(P).all() or not np.isfinite(Z).all():
            raise ValueError('Invalid memory-resident basis archive')
        record.update(offline_cache_key=hashlib.sha256(raw).hexdigest(),offline_cache_hit=True)
        return P,Z,record
    path=Path(folder)/(body_signature(body)+'.npz')
    with np.load(path,allow_pickle=False) as data:
        P=data['Phi'];Z=data['Z'];record=json.loads(str(data['record']))
    if record['body_signature']!=body_signature(body):raise ValueError('Compressed basis belongs to another model')
    if not np.isfinite(P).all() or not np.isfinite(Z).all():raise ValueError('Invalid compressed basis')
    record['offline_cache_key']=hashlib.sha256(path.read_bytes()).hexdigest();record['offline_cache_hit']=True
    return P,Z,record


def cached_compressed_basis(body,policy,folder):
    """One algorithm for every scene; keep only the compact offline artifact."""
    from quasistatic_basis import cached_basis
    from threadpoolctl import threadpool_limits
    if folder is None:raise ValueError('Compression requires an explicit offline cache directory')
    signature=body_signature(body);rank=int(policy['residual_compression_rank']);strategy=policy.get('residual_compression_strategy','traction')
    settings={k:policy.get(k,v) for k,v in DEFAULT_BASIS_POLICY.items()}
    key=hashlib.sha256((signature+str(rank)+strategy+json.dumps(settings,sort_keys=True)).encode()+Path(__file__).read_bytes()).hexdigest()
    path=Path(folder)/('compressed_'+key+'.npz')
    if path.exists():
        with np.load(path,allow_pickle=False) as a:P=a['Phi'];Z=a['Z'];record=json.loads(str(a['record']))
        if record['body_signature']!=signature:raise ValueError('Compressed cache model mismatch')
        record.update(offline_cache_key=key,offline_cache_hit=True,compressed_archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return P,Z,record
    start=time.perf_counter();P,Z,parent=cached_basis(body,policy,folder)
    with threadpool_limits(limits=4):
        C,record=compress_residual(body,P,Z,rank,strategy=strategy)
        record['audit']=audit_compression(body,P,C)
    record.update(body_signature=signature,principal_dimension=P.shape[1],residual_dimension=C.shape[1],free_dimension=len(body.free),
        parent_offline_cache_key=parent['offline_cache_key'],offline_cache_key=key,offline_cache_hit=False,
        no_online_factor_returned=True,basis_bytes=P.nbytes+C.nbytes,preprocessing_seconds=time.perf_counter()-start)
    np.savez_compressed(path,Phi=P,Z=C,record=json.dumps(record))
    record['compressed_archive_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    return P,C,record


