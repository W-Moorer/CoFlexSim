"""Offline physical modal selection and streamed static residual attachment space.

No inverse/factor is returned to the time integrator. Validation loads differ
from construction loads; an unresolved budget raises instead of falling back
to a full-order online solve. All ranks and tolerances belong to solver policy.
"""
import time,json,hashlib
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla


DEFAULT_BASIS_POLICY = dict(candidate_modes=32, maximum_candidates=192,
    principal_modes=6, maximum_principal_modes=96, dynamic_response_tolerance=.05,
    residual_modes=16, maximum_residual_modes=2048, static_response_tolerance=.05,
    surface_landmarks=16, validation_loads=6, basis_seed=714,holdout_response_tolerance=.05,holdout_load_seed=None,
    basis_memory_limit_bytes=256*1024**2,residual_batch_size=8)


class BasisBudgetError(RuntimeError):
    pass


def energy_error(error, reference, metric):
    num=np.maximum(np.sum(error*(metric@error),axis=0),0.)
    den=np.maximum(np.sum(reference*(metric@reference),axis=0),1e-30)
    return float(np.sqrt(num/den).max(initial=0.))


def eigenspace(K,M,count):
    n=K.shape[0]
    if n<=320 or count>=n-1:
        if n>1800:raise BasisBudgetError('Dense complete eigenspace exceeds the offline budget')
        vals,V=la.eigh(K.toarray(),M.toarray());return np.maximum(vals,0.),V
    vals,V=sla.eigsh(K,k=min(count,n-2),M=M,sigma=-1.,which='LM',tol=1e-9,v0=np.linspace(1.,2.,n))
    ids=np.argsort(vals);return np.maximum(vals[ids],0.),V[:,ids]


def elastic_factor(K,M,rigid):
    if not rigid.shape[1]:
        factor=sla.splu(K.tocsc())
        return factor.solve
    MR=sp.csc_matrix(M@rigid)
    factor=sla.splu(sp.bmat([[K,MR],[MR.T,None]],format='csc'))
    def solve(rhs):
        rhs=np.asarray(rhs);vector=rhs.ndim==1;rhs=rhs.reshape(K.shape[0],-1)
        result=factor.solve(np.vstack((rhs,np.zeros((rigid.shape[1],rhs.shape[1])))))[:K.shape[0]]
        return result[:,0] if vector else result
    return solve


def make_channels(body,policy,seed,construction=False):
    """Geometry-derived surface patches; never inspect a reference trajectory."""
    ids=load_support(body)
    if not len(ids):ids=body.free_nodes
    n=len(body.free);xyz=body.X[ids];rng=np.random.default_rng(seed)
    count=min(len(ids),policy['surface_landmarks'] if construction else policy['validation_loads'])
    if construction:
        chosen=[0];distance=la.norm(xyz-xyz[0],axis=1)
        for _ in range(count-1):
            j=int(np.argmax(distance));chosen.append(j);distance=np.minimum(distance,la.norm(xyz-xyz[j],axis=1))
    else:chosen=rng.choice(len(ids),count,replace=False).tolist()
    node_to_free=np.full(len(body.X),-1,int);node_to_free[body.free_nodes]=np.arange(len(body.free_nodes))
    columns=[]
    if construction:
        # Include prescribed load patterns FIRST, before truncating landmark
        # snapshots. Rotation closure spans f, axis cross f, axis cross(axis cross f).
        nodal=np.asarray(body.spec.get('nodal_force',np.zeros_like(body.X)),float)
        accel=np.asarray(body.spec.get('body_acceleration',[0.,0.,0.]),float)
        nodal=nodal+body.native.mass[:,None]*accel
        patterns=[nodal]
        if body.revolute:patterns.extend([np.cross(body.axis,nodal),np.cross(body.axis,np.cross(body.axis,nodal))])
        for field in patterns:
            v=field.ravel()[body.free]
            if la.norm(v)>0:columns.append(v/la.norm(v))
        if body.reference:
            tangent=(np.cross(body.axis,body.X-body.center) if body.revolute else np.tile(body.axis,(len(body.X),1))).ravel()
            inertial=(body.mass*tangent)[body.free]
            if la.norm(inertial)>0:columns.append(inertial/la.norm(inertial))
    for j in chosen:
        near=np.argsort(la.norm(xyz-xyz[j],axis=1))[:min(3,len(ids))]
        for direction in (np.eye(3) if construction else [rng.normal(size=3)]):
            direction=direction/la.norm(direction);v=np.zeros(n)
            for node in ids[near]:v[3*node_to_free[node]:3*node_to_free[node]+3]=direction/np.sqrt(len(near))
            columns.append(v)
    return np.column_stack(columns)


def load_support(body):
    """Declared external load/contact domain; every response still uses whole K.

    A non-contact scene can declare a load support instead of auditing arbitrary
    volume point loads. This narrows the advertised load family, not the body.
    """
    prescribed=body.spec.get('correction_load_nodes')
    if prescribed is not None:
        ids=np.asarray(prescribed,dtype=int)
        if np.any(ids<0) or np.any(ids>=len(body.X)):raise ValueError('Invalid correction load nodes')
        # Include all declared contact and nonzero external-load nodes as well.
        force=np.asarray(body.spec.get('nodal_force',np.zeros_like(body.X)))
        ids=np.unique(np.r_[ids,np.unique(body.spec.get('contact_faces',[])),np.flatnonzero(np.any(force!=0,axis=1))])
    else:ids=np.unique(body.spec.get('contact_faces',body.faces))
    ids=np.intersect1d(ids,body.free_nodes).astype(int)
    return ids if len(ids) else body.free_nodes


def physical_reference(body):
    K,M=body.K,body.M
    if not body.reference:return K,M,None
    tangent=(np.cross(body.axis,body.X-body.center) if body.revolute else np.tile(body.axis,(len(body.X),1))).ravel()
    c=(body.mass*tangent)[body.free];a=float(np.dot(body.mass,tangent*tangent))
    return sp.block_diag((sp.csc_matrix((1,1)),K),format='csc'),sp.bmat([[sp.csc_matrix([[a]]),sp.csc_matrix(c[None])],[sp.csc_matrix(c[:,None]),M]],format='csc'),tangent[body.free]


def build_basis(body,overrides=None):
    policy=DEFAULT_BASIS_POLICY|{k:v for k,v in dict(overrides or {}).items() if k in DEFAULT_BASIS_POLICY};start=time.perf_counter()
    K,M=body.K,body.M;n=K.shape[0];seed=policy['basis_seed']
    if n==0:raise ValueError('Body has no elastic degrees of freedom')
    train=make_channels(body,policy,seed,True)
    validate=make_channels(body,policy,seed+1)
    audit=make_channels(body,policy,seed+2)
    all_checks=np.c_[validate,audit]
    Kgen,Mgen,tangent=physical_reference(body)
    rhsgen=np.vstack((tangent@all_checks,all_checks)) if body.reference else all_checks
    if body.reference:rhsgen=np.c_[rhsgen,np.r_[1.,np.zeros(n)]]
    bands=np.unique(np.maximum(np.asarray(body.spec.get('frequency_band_hz',[0.,50.,100.,200.])),0.))
    probes=np.unique(np.r_[bands,(bands[:-1]+bands[1:])*.5])
    if body.reference or not len(body.fixed):probes=probes[probes>0]
    if not len(probes):raise BasisBudgetError('At least one positive frequency probe is required for a body with reference/rigid motion')
    exact=[]
    for f in probes:
        w=2*np.pi*f
        try:full=sla.splu((Kgen-w*w*Mgen).tocsc()).solve(rhsgen)
        except RuntimeError as e:raise BasisBudgetError(f'Frequency probe {f} Hz is singular; define a finite-time audit') from e
        exact.append((w,full,(Kgen+w*w*Mgen).tocsc()))
    pool=min(policy['candidate_modes'],n);selection_history=[]
    while True:
        vals,V=eigenspace(K,M,pool);rigid=vals<=max(1e-8,float(vals.max())*1e-11)
        inv=elastic_factor(K,M,V[:,rigid]);static=inv(all_checks)
        flex=np.flatnonzero(~rigid);rigid_ids=np.flatnonzero(rigid)
        freq=np.sqrt(vals)/(2*np.pi)
        bandmax=float(bands.max(initial=0.));protected=set(rigid_ids.tolist())
        protected.update(np.flatnonzero(freq<=bandmax*1.05).tolist())
        participation=la.norm(V.T@train,axis=1)
        denom=np.min(np.abs(vals[:,None]-(2*np.pi*bands[None])**2),axis=1)
        score=participation*np.sqrt(np.maximum(vals,1.))/(denom+1e-12*max(1.,float(vals.max())))
        order=np.argsort(score)[::-1]
        count=max(policy['principal_modes'],len(protected));passed=False
        while count<=min(len(vals),policy['maximum_principal_modes']):
            ids=set(order[:count].tolist())|protected
            for i in list(ids):ids.update(np.flatnonzero(abs(vals-vals[i])<=1e-6*max(1.,vals[i])).tolist())
            ids=np.array(sorted(ids),int);Phi=V[:,ids]
            if len(ids)>policy['maximum_principal_modes']:break
            elastic_ids=ids[~rigid[ids]];Pe=V[:,elastic_ids]
            residual=static-Pe@((Pe.T@all_checks)/vals[elastic_ids,None])
            VV=la.block_diag(np.ones((1,1)),Phi) if body.reference else Phi
            Mr=VV.T@(Mgen@VV);Kr=VV.T@(Kgen@VV);errors=[]
            for w,full,metric in exact:
                try:a=la.solve(Kr-w*w*Mr,VV.T@rhsgen,assume_a='sym')
                except la.LinAlgError:errors.append(float('inf'));continue
                approx=VV@a
                if body.reference:approx[1:,:all_checks.shape[1]]+=residual
                else:approx+=residual
                errors.append(energy_error(approx-full,full,metric))
            err=max(errors,default=0.)
            selection_history.append({'pool':len(vals),'retained':len(ids),'maximum_dynamic_energy_error':err})
            if err<=policy['dynamic_response_tolerance']:passed=True;break
            if count==min(len(vals),policy['maximum_principal_modes']):break
            count=min(len(vals),policy['maximum_principal_modes'],max(count+1,count*2))
        if passed:break
        if pool>=min(n,policy['maximum_candidates']):raise BasisBudgetError('Principal mode response audit failed within candidate/rank budget: '+str(selection_history[-1] if selection_history else {'required_minimum':count,'candidate_pool':len(vals)}))
        pool=min(n,policy['maximum_candidates'],max(pool+1,pool*2))
    # Store only the retained modes; full eigenvectors and sparse factors are offline.
    elastic_ids=ids[~rigid[ids]];Pe=V[:,elastic_ids];evalues=vals[elastic_ids]
    def residual_action(rhs):return inv(rhs)-Pe@((Pe.T@rhs)/evalues[:,None])
    Z=np.zeros((n,0));validation_static=static;residual_history=[]
    channels=train;budget=min(policy['maximum_residual_modes'],n-Phi.shape[1])
    # Phi/Z and the concatenated reconstruction matrix coexist in the backend.
    max_cols=max(0,int(policy['basis_memory_limit_bytes']//max(1,16*n))-Phi.shape[1])
    budget=min(budget,max_cols)
    def orthogonal_append(Y):
        nonlocal Z
        # K-complement is mathematically exact for elastic eigenvectors; remove
        # numerical rigid components in the M metric as well.
        if V[:,rigid].shape[1]:Y-=V[:,rigid]@(V[:,rigid].T@(M@Y))
        if Pe.shape[1]:Y-=Pe@((Pe.T@(K@Y))/evalues[:,None])
        for _ in range(2):
            if Z.shape[1]:Y-=Z@(Z.T@(K@Y))
        gram=Y.T@(K@Y);d,U=la.eigh((gram+gram.T)*.5)
        keep=np.flatnonzero(d>max(1e-22,float(d.max(initial=0.))*1e-10))[::-1]
        keep=keep[:max(0,budget-Z.shape[1])]
        if len(keep):Z=np.c_[Z,Y@U[:,keep]/np.sqrt(d[keep])[None]]
    for step in range(0,channels.shape[1],8):
        orthogonal_append(residual_action(channels[:,step:step+8]))
        if Z.shape[1]>=min(policy['residual_modes'],budget):break
    # Adaptive construction load set is separate from both validation sets.
    rng=np.random.default_rng(seed+11)
    batch_size=int(policy['residual_batch_size'])
    if batch_size<1:raise ValueError('residual_batch_size must be positive')
    max_batches=max(1,(budget+batch_size-1)//batch_size)
    surface=load_support(body)
    if not len(surface):surface=body.free_nodes
    free_index=np.full(len(body.X),-1,int);free_index[body.free_nodes]=np.arange(len(body.free_nodes))
    surface_dofs=(3*free_index[surface,None]+np.arange(3)).ravel()
    for batch in range(max_batches+1):
        approx=Pe@((Pe.T@all_checks)/evalues[:,None])
        if Z.shape[1]:approx+=Z@(Z.T@all_checks)
        err=energy_error(approx-validation_static,validation_static,K)
        residual_history.append({'rank':Z.shape[1],'maximum_static_energy_error':err})
        if err<=policy['static_response_tolerance']:break
        if Z.shape[1]>=budget or batch==max_batches:raise BasisBudgetError('Surface residual response audit failed within memory/rank budget: '+str(residual_history[-1]))
        B=np.zeros((n,min(batch_size,budget-Z.shape[1])))
        B[surface_dofs]=rng.normal(size=(len(surface_dofs),B.shape[1]))/np.sqrt(len(surface_dofs))
        previous=Z.shape[1];orthogonal_append(residual_action(B))
        if Z.shape[1]==previous:raise BasisBudgetError('Residual construction stagnated before response target')
    # K-normalization is good for compression; rescale to M-normalized columns
    # before PD to avoid ill-conditioned generalized coordinates.
    if Z.shape[1]:Z/=np.sqrt(np.sum(Z*(M@Z),axis=0))[None]
    # Untouched third load set: report the ACTUAL compressed massless model,
    # not an exact-residual proxy. These loads never select/enrich the basis.
    holdout_seed=seed+3 if policy.get('holdout_load_seed') is None else int(policy['holdout_load_seed'])
    if holdout_seed in (seed,seed+1,seed+2,seed+11):raise ValueError('Holdout seed overlaps construction/selection samples')
    held=make_channels(body,policy,holdout_seed);heldgen=np.vstack((tangent@held,held)) if body.reference else held
    if body.reference:heldgen=np.c_[heldgen,np.r_[1.,np.zeros(n)]]
    # Condense the algebraic block ONCE. Forming a giant KK-w^2 MM and
    # refactoring it at every probe wastes both storage and cubic work.
    VD=la.block_diag(np.ones((1,1)),Phi) if body.reference else Phi
    kd=VD.T@(Kgen@VD);md=VD.T@(Mgen@VD);rd=VD.T@heldgen
    coupling=Phi.T@(K@Z)
    if body.reference:coupling=np.vstack((np.zeros((1,Z.shape[1])),coupling))
    if Z.shape[1]:
        static_factor=la.cho_factor(Z.T@(K@Z),check_finite=False)
        response=la.cho_solve(static_factor,Z.T@held,check_finite=False)
        if body.reference:response=np.c_[response,np.zeros(Z.shape[1])]
        cross=la.cho_solve(static_factor,coupling.T,check_finite=False)
    else:response=np.zeros((0,heldgen.shape[1]));cross=np.zeros((0,VD.shape[1]))
    schur=kd-coupling@cross;right=rd-coupling@response
    hold_errors=[]
    for w,_,metric in exact:
        full=sla.splu((Kgen-w*w*Mgen).tocsc()).solve(heldgen)
        a=la.solve(schur-w*w*md,right,assume_a='sym');approx=VD@a
        correction=Z@(response-cross@a)
        if body.reference:approx[1:]+=correction
        else:approx+=correction
        hold_errors.append(energy_error(approx-full,full,metric))
    if max(hold_errors,default=0.)>policy['holdout_response_tolerance']:
        raise BasisBudgetError('Untouched compressed-model response audit failed: '+str(max(hold_errors)))
    record={'construction':'physical_K_static_residual_surface_loads','principal_dimension':Phi.shape[1],
        'residual_dimension':Z.shape[1],'free_dimension':n,'selected_indices':ids.tolist(),
        'candidate_frequencies_hz':freq.tolist(),'selection_history':selection_history,
        'static_validation_history':residual_history,'validation_load_seed':seed+1,'audit_load_seed':seed+2,
        'dynamic_probes_hz':probes.tolist(),'reference_point_mass_coupling_in_audit':body.reference,
        'untouched_holdout_seed':holdout_seed,'compressed_model_holdout_energy_errors':hold_errors,
        'compressed_model_holdout_maximum_energy_error':max(hold_errors,default=0.),
        'basis_bytes':Phi.nbytes+Z.nbytes,'persistent_basis_array_bytes':2*(Phi.nbytes+Z.nbytes),'memory_limit_bytes':policy['basis_memory_limit_bytes'],
        'no_online_factor_returned':True,'preprocessing_seconds':time.perf_counter()-start,'policy':policy}
    return Phi.copy(),Z.copy(),record


def cached_basis(body,policy,folder=None):
    """Content-addressed offline artifact. Never cache a full inverse or LU."""
    if folder is None:return build_basis(body,policy)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(Path(__file__).read_bytes())
    relevant=DEFAULT_BASIS_POLICY|{k:v for k,v in policy.items() if k in DEFAULT_BASIS_POLICY}
    digest.update(json.dumps(relevant,sort_keys=True).encode())
    for key in sorted(body.spec):
        value=body.spec[key]
        digest.update(key.encode())
        if isinstance(value,np.ndarray):digest.update(str((value.shape,value.dtype)).encode());digest.update(value.tobytes())
        else:digest.update(json.dumps(value,sort_keys=True).encode())
    # Include the assembled operators, so material-kernel changes invalidate.
    for A in (body.K,body.M):
        for v in (A.data,A.indices,A.indptr):digest.update(v.tobytes())
    key=digest.hexdigest();path=folder/(key+'.npz')
    if path.exists():
        with np.load(path,allow_pickle=False) as archive:
            Phi=archive['Phi'];Z=archive['Z'];record=json.loads(str(archive['record']))
        if Phi.shape[0]!=len(body.free) or Z.shape[0]!=len(body.free) or not np.all(np.isfinite(np.c_[Phi,Z])):raise ValueError('Corrupt offline basis artifact')
        record['offline_cache_hit']=True;record['offline_cache_key']=key;return Phi,Z,record
    Phi,Z,record=build_basis(body,policy);record['offline_cache_hit']=False;record['offline_cache_key']=key
    temporary=path.with_suffix('.pending.npz');np.savez_compressed(temporary,Phi=Phi,Z=Z,record=json.dumps(record));temporary.replace(path)
    return Phi,Z,record
