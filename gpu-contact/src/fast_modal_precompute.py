"""Equivalent offline preparation; mode selection changes remain opt-in.

The original numerical sources remain frozen. The selector keeps the original
cutoff, reuses one shifted factor and accepts a same-model candidate-count hint.
The runtime builder canonicalizes once, projects in free coordinates, and derives
Cartesian inertia blocks from already available reduced integrals.
"""
from types import SimpleNamespace
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
from threadpoolctl import threadpool_limits
from surface_cms import cross_matrix


def select_modes(body,dt,alpha=-.05,relative_mass_limit=.05,candidate_hint=None,threads=4):
    n=body.K.shape[0];beta=(1-alpha)**2/4;c=1/(beta*dt**2*(1+alpha))
    threshold=c/relative_mass_limit;k=min(int(candidate_hint or 512),n-2)
    if k<1 or k>2048 or threads<1:raise ValueError('Require 1--2048 candidates and positive thread count')
    start=time.perf_counter();attempts=[];sigma=-1e-3
    with threadpool_limits(threads):
        tick=time.perf_counter();factor=sla.splu((body.K-sigma*body.M).tocsc())
        factor_s=time.perf_counter()-tick
        inverse=sla.LinearOperator(body.K.shape,matvec=factor.solve,matmat=factor.solve,dtype=np.float64)
        while True:
            tick=time.perf_counter()
            ev,P=sla.eigsh(body.K,k=k,M=body.M,sigma=sigma,which='LM',OPinv=inverse,
                tol=1e-9,v0=np.linspace(1.,2.,n))
            order=np.argsort(ev);ev=ev[order];P=P[:,order]
            attempts.append(dict(candidates=k,seconds=time.perf_counter()-tick))
            if ev[0]<=0:raise ValueError('Elastic rigid modes must be removed')
            count=int(np.searchsorted(ev,threshold*(1+1e-5)))
            if count:
                while count<len(ev) and abs(ev[count]-ev[count-1])<=1e-5*ev[count-1]:count+=1
            if count<len(ev):break
            if k>=min(2048,n-2):raise ValueError('Cutoff not bracketed; no silent mode truncation')
            k=min(2*k,2048,n-2)
        P=P[:,:count]
        kp=body.K@P;mp=body.M@P
        residual=np.linalg.norm(kp-mp*ev[:count],axis=0)/np.maximum(
            np.linalg.norm(kp,axis=0)+np.linalg.norm(mp*ev[:count],axis=0),1e-30)
        orth=float(la.norm(P.T@mp-np.eye(count))/max(np.sqrt(count),1))
        worst=float(residual.max(initial=0.))
        if worst>5e-7 or orth>1e-7:raise ValueError(('Eigenpair acceptance failed',worst,orth))
    return P,dict(criterion='discrete_static_tail_inertia_bound_v1',alpha=alpha,dt_s=dt,
        relative_mass_limit=relative_mass_limit,cutoff_hz=float(np.sqrt(threshold)/(2*np.pi)),
        selected_modes=count,candidate_modes=k,lowest_omitted_hz=float(np.sqrt(ev[count])/(2*np.pi)),
        actual_fixed_frame_mass_ratio=float(c/ev[count]),offline_seconds=time.perf_counter()-start,
        nonlinear_trajectory_error_bound=False,shifted_factorizations=1,factorization_s=factor_s,
        attempts=attempts,blas_threads=threads,max_eigenpair_relative_residual=worst,mass_orthogonality_relative=orth)


def prepare_runtime(fe,principal,threads=4):
    """Return surface runtime and canonical free-coordinate basis (temporary)."""
    tick=time.perf_counter();spec=fe.spec
    if len(fe.free)%3 or not np.array_equal(fe.free.reshape(-1,3)%3,np.tile(np.arange(3),(len(fe.free)//3,1))):
        raise ValueError('Fast preparation currently requires whole-node elastic constraints')
    if not np.all(fe.free.reshape(-1,3)//3 == fe.free.reshape(-1,3)[:,:1]//3):
        raise ValueError('Free elastic coordinates must be grouped by node')
    for key in ('body_acceleration','nodal_force','initial_displacement','initial_velocity'):
        if np.any(spec.get(key,0)):raise ValueError('Load/initial channel requires explicit preparation: '+key)
    with threadpool_limits(threads):
        p0=np.asarray(principal);k0=p0.T@(fe.K@p0);m0=p0.T@(fe.M@p0)
        ev,q=la.eigh(k0,m0);p=np.ascontiguousarray(p0@q)
        # No second volume projection after the basis transform.
        k=q.T@k0@q;m=q.T@m0@q
        nodes=np.unique(spec['contact_faces']);srows=(3*nodes[:,None]+np.arange(3)).ravel()
        inverse=np.full(fe.X.size,-1,dtype=np.int64);inverse[fe.free]=np.arange(len(fe.free))
        lookup=inverse[srows];valid=lookup>=0;ps=np.zeros((len(srows),p.shape[1]));ps[valid]=p[lookup[valid]]
        c=cross_matrix(fe.axis);x=(fe.X-fe.center).ravel();cx=(x.reshape(-1,3)@c.T).ravel()
        w=fe.mass[fe.free];xf=x[fe.free];cxf=cx[fe.free]
        cp=np.einsum('ij,njk->nik',c,p.reshape(len(fe.free)//3,3,p.shape[1])).reshape(p.shape)
        qr=p.T@(w[:,None]*cp);dr=-(cp.T@(w[:,None]*cp))
        c0=p.T@(w*cxf);d0=-(cp.T@(w*cxf));i0=float(cx@(fe.mass*cx))
        mt=p.T@(w*np.tile(fe.axis,len(fe.free)//3))
        face_lookup=np.full(len(fe.X),-1,dtype=int);face_lookup[nodes]=np.arange(len(nodes))
        small={key:spec[key] for key in ('torque','reference_force','initial_angle','initial_angular_velocity',
            'initial_reference_position','initial_reference_velocity','guide_stiffness','guide_damping',
            'guide_rest_coordinate','guide_target_velocity') if key in spec}
        small['contact_faces']=face_lookup[spec['contact_faces']]
        b=SimpleNamespace(X=fe.X[nodes].copy(),spec=small,center=fe.center.copy(),axis=fe.axis.copy(),
            reference=fe.reference,revolute=fe.revolute,prismatic=fe.prismatic,offset=fe.offset,
            P=ps,K=k,M=m,Q=qr,D=dr,c0=c0,d0=d0,I0=i0,mt=mt,total_mass=float(fe.native.mass.sum()),R=None,
            rd=fe.offset+p.shape[1],original_nodes=nodes,original_surface_rows=srows,kind='residual')
        if b.revolute:
            nm=p.shape[1];h0=np.zeros((nm+1,nm+1));h1=h0.copy();h2=h0.copy()
            r=p.T@(w*xf);free_i=float(cxf@(w*cxf))
            h0[0,0]=xf@(w*xf);h0[0,1:]=r;h0[1:,0]=r;h0[1:,1:]=m
            h1[1:,0]=c0;h1[0,1:]=-c0;h1[1:,1:]=qr
            h2[0,0]=-free_i;h2[1:,0]=d0;h2[0,1:]=d0;h2[1:,1:]=dr
            b.cartesian_inertia=dict(H0=h0,H1=h1,H2=h2,Minv=la.inv(m),hub_I=i0-free_i)
        b.record=dict(kind='residual',full_elastic_dofs=len(fe.free),modal_dofs=p.shape[1],surface_dofs=len(srows),
            offline_seconds=time.perf_counter()-tick,canonicalizations=1,cartesian_gram_reused=True,
            residual_source='S K^-1 S^T - S Phi (Phi^T K Phi)^-1 Phi^T S^T',
            original_author_executable=False,equivalent_preparation=True)
    return b,p


def restrict_runtime(body,ids):
    """Change only retained dynamic modes; omitted static compliance is separate."""
    import copy
    ids=np.asarray(ids,dtype=int)
    if len(np.unique(ids))!=len(ids):raise ValueError('Repeated mode')
    b=copy.copy(body);b.P=body.P[:,ids].copy();b.rd=b.offset+len(ids)
    for key in ('K','M','Q','D'):setattr(b,key,getattr(body,key)[np.ix_(ids,ids)].copy())
    for key in ('c0','d0','mt'):setattr(b,key,getattr(body,key)[ids].copy())
    if hasattr(body,'cartesian_inertia'):
        ix=np.r_[0,ids+1];old=body.cartesian_inertia
        b.cartesian_inertia={k:old[k][np.ix_(ix,ix)].copy() for k in ('H0','H1','H2')}
        b.cartesian_inertia.update(Minv=la.inv(b.M),hub_I=old['hub_I'])
    b.record={**body.record,'modal_dofs':len(ids),'mode_selection_changed':True}
    return b


class RetainedSpaceResidual:
    """Reuse one library: R_new = R_old + P_old K_old^-1 P_old^T - P_new K_new^-1 P_new^T.

    No second dense residual library. The update preserves whole-body static
    compliance including cross-region terms; it does not preserve omitted inertia.
    """
    sealed=True
    def __init__(self,source,old,new):
        self.source=source;self.n=source.n;self.old=old.P;self.new=new.P
        cache=getattr(source,'retained_reference_cache',None)
        if cache is None:
            cache=(old.P,old.K,la.solve(old.K,old.P.T,assume_a='pos'))
            source.retained_reference_cache=cache
        if cache[0] is not old.P or cache[1] is not old.K:raise ValueError('Library modal reference differs')
        self.old_right=cache[2];self.new_right=la.solve(new.K,new.P.T,assume_a='pos')
        self.statistics=dict(source='same_bank_retained_space_correction',online_factor_solves=0,
            stored_bytes=source.statistics.get('stored_bytes',0),mode_selection_changed=True)
    def block(self,rows,cols):
        rows=np.asarray(rows);cols=np.asarray(cols)
        return (self.source.block(rows,cols)+self.old[rows]@self.old_right[:,cols]
                -self.new[rows]@self.new_right[:,cols])
    def columns(self,cols):return self.block(np.arange(self.n),cols)
    def apply(self,force,rows=None):
        rows=np.arange(self.n) if rows is None else rows;ids=np.flatnonzero(force)
        return self.block(rows,ids)@force[ids]


def materialize_retained_space(library,old,new,threads=4):
    """Offline-only retargeting of ONE private RAM bank, without a second bank.

    Call only after destroying its previous simulator. Canonical on-disk banks
    are explicitly rejected. The retained-space correction is paid once here,
    not on every online block query. A failed update leaves the bank unsealed.
    """
    if getattr(library,'backing',None) is not None or 'file_path' in library.statistics:
        raise ValueError('Do not mutate the canonical disk library')
    if not library.sealed or library.n!=len(old.P) or library.n!=len(new.P):
        raise ValueError('Invalid private RAM bank or surface identity')
    identity=getattr(library,'retained_body_identity',old)
    if identity is not old:raise ValueError('Previous retained space differs')
    tick=time.perf_counter();probe=np.unique(np.linspace(0,library.n-1,19,dtype=int))
    with threadpool_limits(threads):
        old_right=la.solve(old.K,old.P.T,assume_a='pos')
        new_right=la.solve(new.K,new.P.T,assume_a='pos')
        expected=library.block(probe,probe)+old.P[probe]@old_right[:,probe]
        library.sealed=False;library.statistics['sealed']=False
        bounds=np.zeros(library.n);tile_size=library.tile_size
        for (a,b),tile in library.tiles.items():
            rows=slice(a*tile_size,a*tile_size+tile.shape[0]);cols=slice(b*tile_size,b*tile_size+tile.shape[1])
            delta=old.P[rows]@old_right[:,cols]-new.P[rows]@new_right[:,cols]
            tile.flags.writeable=True
            try:
                tile+=delta
                if a==b:tile[:]=(tile+tile.T)*.5
                if not np.isfinite(tile).all():raise ValueError('Nonfinite retargeted tile')
                bounds[rows]=np.maximum(bounds[rows],abs(tile).max(axis=1))
                bounds[cols]=np.maximum(bounds[cols],abs(tile).max(axis=0))
            finally:tile.flags.writeable=False
        actual=library.block(probe,probe)+new.P[probe]@new_right[:,probe]
        error=float(la.norm(actual-expected)/max(la.norm(expected),1e-30))
        if error>1e-10:raise ValueError(('Static compliance changed',error))
    library.row_bound=bounds
    if hasattr(library,'far_column_bound'):del library.far_column_bound
    library.retained_body_identity=new;library.sealed=True
    elapsed=time.perf_counter()-tick
    library.statistics.update(sealed=True,maximum_absolute_entry=float(bounds.max(initial=0.)),
        offline_s=library.statistics['offline_s']+elapsed,retained_retarget_s=elapsed,
        block_requests=0,query_s=0.)
    return dict(offline_s=elapsed,static_compliance_relative=error,old_modes=old.P.shape[1],
        new_modes=new.P.shape[1],stored_bytes=library.statistics['stored_bytes'],second_bank_allocated=False)
