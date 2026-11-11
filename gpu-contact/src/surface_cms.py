"""Traditional modal dynamics with load-coordinate residual surface flexibility.

Independent literature-based baseline, not an author executable and not PD.
Small elastic strain in an exact one-axis floating reference frame. HHT acts on
the retained dynamic equations; the residual response is algebraic at the current
endpoint. Only the declared contact surface exists in the online geometry.
"""
from types import SimpleNamespace
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from coflex_core import rotation
from contact_candidate_cache import ContactCandidateCache
from consistent_mortar_contact import ConsistentMortarContact
from contact_multiplier_history import ContactMultiplierHistory
from shared_contact import solve_working_qp


def dense(a):
    return a.toarray() if sp.issparse(a) else np.asarray(a)


def cross_matrix(axis):
    x,y,z=axis
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def select_principal_for_static_tail(body,dt,alpha=-.05,relative_mass_limit=.05):
    """Retain every mode whose omitted inertia is not small in the global step.

    For a fixed-frame eigenmode, H=K+cM and static instead of H inverse has
    response error c/omega^2 relative to H inverse. This is an operator-based
    bandwidth selection, not a guarantee on nonlinear contact trajectories.
    """
    n=body.K.shape[0];beta=(1-alpha)**2/4;c=1/(beta*dt**2*(1+alpha))
    threshold=c/relative_mass_limit;k=min(512,n-2);start=time.perf_counter()
    while True:
        ev,P=sla.eigsh(body.K,k=k,M=body.M,sigma=-1e-3,which='LM',tol=1e-9,
                       v0=np.linspace(1.,2.,n))
        order=np.argsort(ev);ev=ev[order];P=P[:,order]
        if ev[0]<=0:raise ValueError('Eliminate elastic rigid modes before residual-flexibility construction')
        if ev[-1]>threshold*(1+1e-5):break
        if k>=min(2048,n-2):raise ValueError('Requested static-tail bound needs a larger eigenspace; no silent truncation')
        k=min(2*k,2048,n-2)
    count=int(np.searchsorted(ev,threshold*(1+1e-5)))
    # Include the entire boundary eigenvalue cluster.
    if count:
        while count<len(ev) and abs(ev[count]-ev[count-1])<=1e-5*ev[count-1]:count+=1
    selected=P[:,:count]
    return selected,dict(criterion='discrete_static_tail_inertia_bound_v1',alpha=alpha,dt_s=dt,
        relative_mass_limit=relative_mass_limit,cutoff_hz=float(np.sqrt(threshold)/(2*np.pi)),
        selected_modes=count,candidate_modes=k,lowest_omitted_hz=float(np.sqrt(ev[count])/(2*np.pi)),
        actual_fixed_frame_mass_ratio=float(c/ev[count]),offline_seconds=time.perf_counter()-start,
        nonlinear_trajectory_error_bound=False)


def prepare_body(body, principal, kind='residual', batch=64):
    """All volume work and K solves occur here, before the timed simulation.

    `full` keeps every free elastic FE coordinate and is an internal consistency
    reference. `modal` omits the residual term. `residual` uses its exact surface
    trace, not the previously truncated static attachment basis.
    """
    tick=time.perf_counter();b=body;spec=b.spec
    if kind not in ('residual','modal','full'):raise ValueError(kind)
    if np.any(spec.get('body_acceleration',0)) or np.any(spec.get('nodal_force',0)):
        raise ValueError('This release supports reference torque/force and guide loads; distributed load channels need offline preparation')
    if np.any(spec.get('initial_displacement',0)) or np.any(spec.get('initial_velocity',0)):
        raise ValueError('Nonzero elastic initial states must be projected explicitly')
    nodes=np.unique(spec['contact_faces']);srows=(3*nodes[:,None]+np.arange(3)).ravel()
    inv=np.full(b.X.size,-1,int);inv[b.free]=np.arange(len(b.free));lookup=inv[srows]
    valid=np.flatnonzero(lookup>=0);ns=len(srows);n=len(b.free)
    P=sp.eye(n,format='csr') if kind=='full' else np.asarray(principal)
    if kind!='full':
        # Canonicalize the supplied, load-selected span, without replacing it
        # with a new low-frequency-only selection.
        ev,Q=la.eigh(P.T@(b.K@P),P.T@(b.M@P));P=np.ascontiguousarray(P@Q)
    K=P.T@(b.K@P);M=P.T@(b.M@P)
    S=sp.csr_matrix((np.ones(len(valid)),(valid,lookup[valid])),shape=(ns,n))
    Ps=S@P
    R=None;audits={}
    if kind=='residual':
        fac=sla.splu(b.K.tocsc());R=np.zeros((ns,ns))
        for j in range(0,ns,batch):
            ids=np.arange(j,min(j+batch,ns));rhs=S[ids].T.toarray()
            R[:,ids]=S@fac.solve(rhs)
        modal=Ps@la.solve(K,Ps.T,assume_a='pos');R-=modal;R=(R+R.T)*.5
        rng=np.random.default_rng(20261105);f=rng.normal(size=(ns,8))
        exact=S@fac.solve(S.T@f);restored=Ps@la.solve(K,Ps.T@f,assume_a='pos')+R@f
        audits['static_surface_reconstruction_relative']=float(la.norm(restored-exact)/la.norm(exact))
        audits['surface_reciprocity_relative']=float(la.norm(R-R.T)/max(la.norm(R),1e-30))
        audits['sampled_residual_work_min']=float(np.min(np.sum(f*(R@f),axis=0)))
        del fac
    # Exact floating-frame inertia integrals. Fixed/hub nodes still contribute
    # to reference inertia, although their elastic shape functions vanish.
    C=cross_matrix(b.axis);Cfull=sp.kron(sp.eye(len(b.X),format='csr'),sp.csr_matrix(C),format='csr')
    E=sp.csr_matrix((np.ones(n),(b.free,np.arange(n))),shape=(b.X.size,n))
    V=E@P;weighted=sp.diags(b.mass);X=(b.X-b.center).ravel();CV=Cfull@V;CX=Cfull@X
    Q=V.T@(weighted@CV);D=-(CV.T@(weighted@CV))
    c0=np.asarray(V.T@(b.mass*CX)).ravel();d0=-np.asarray(CV.T@(b.mass*CX)).ravel()
    I0=float(np.dot(CX,b.mass*CX));translation=np.tile(b.axis,len(b.X))
    mt=np.asarray(V.T@(b.mass*translation)).ravel();total=float(b.native.mass.sum())
    face_lookup=np.full(len(b.X),-1,int);face_lookup[nodes]=np.arange(len(nodes))
    small_spec={k:spec[k] for k in ('torque','reference_force','initial_angle','initial_angular_velocity',
        'initial_reference_position','initial_reference_velocity','guide_stiffness','guide_damping',
        'guide_rest_coordinate','guide_target_velocity') if k in spec}
    small_spec['contact_faces']=face_lookup[spec['contact_faces']]
    runtime=SimpleNamespace(X=b.X[nodes].copy(),spec=small_spec,center=b.center.copy(),axis=b.axis.copy(),
        reference=b.reference,revolute=b.revolute,prismatic=b.prismatic,offset=b.offset,
        P=Ps,K=K,M=M,Q=Q,D=D,c0=c0,d0=d0,I0=I0,mt=mt,total_mass=total,R=R,
        rd=b.offset+P.shape[1],original_nodes=nodes,original_surface_rows=srows,kind=kind,
        record=dict(kind=kind,full_elastic_dofs=n,modal_dofs=P.shape[1],surface_dofs=ns,
                    audits=audits,offline_seconds=time.perf_counter()-tick,
                    residual_source='S K^-1 S^T - S Phi (Phi^T K Phi)^-1 Phi^T S^T',
                    elastic_boundary='original_fixed_and_hub_dofs_eliminated',
                    original_author_executable=False))
    return runtime


def inertia(b,q,v):
    """Return mass matrix, reference/modal coupling, and convective inertia."""
    a=q[b.offset:];ad=v[b.offset:];bias=np.zeros(b.rd)
    if not b.reference:return b.M,None,None,bias
    if b.prismatic:return b.M,b.mt,b.total_mass,bias
    w=v[0];Da=np.asarray(b.D@a).ravel();cross=b.c0+np.asarray(b.Q@a).ravel()
    moment=b.I0-2*b.d0@a-a@Da
    bias[1:]=2*w*np.asarray(b.Q@ad).ravel()+w*w*(b.d0+Da)
    bias[0]=-2*w*np.dot(b.d0+Da,ad)
    return b.M,cross,float(moment),bias


class DynamicBlock:
    def __init__(self,b,factor):
        self.b=b;self.factor=factor;self.base=b.K+factor*b.M
        self.solve_elastic=(sla.splu(self.base.tocsc()).solve if sp.issparse(self.base)
                            else lambda x,cf=la.cho_factor(self.base,check_finite=False):la.cho_solve(cf,x,check_finite=False))
    def update(self,q,v,guide=0.):
        M,c,I,bias=inertia(self.b,q,v);self.c=c;self.I=I;self.bias=bias
        if self.b.reference:
            self.h=self.factor*c;self.u=self.solve_elastic(self.h)
            self.schur=self.factor*I+guide-self.h@self.u
            if self.schur<=0:raise ValueError('Nonpositive floating-frame inertia Schur complement')
        self.guide=guide
    def solve(self,rhs):
        o=self.b.offset;y=self.solve_elastic(rhs[o:])
        if not o:return y
        theta=(rhs[0]-self.h@y)/self.schur
        return np.concatenate((np.asarray(theta)[None],y-self.u.reshape((-1,)+(1,)*(rhs.ndim-1))*theta),axis=0)
    def mass_action(self,a):
        if not self.b.reference:return np.asarray(self.b.M@a).ravel()
        return np.r_[self.I*a[0]+self.c@a[1:],self.c*a[0]+np.asarray(self.b.M@a[1:]).ravel()]
    def action(self,a):
        out=self.factor*self.mass_action(a);out[self.b.offset:]+=np.asarray(self.b.K@a[self.b.offset:]).ravel()
        if self.b.offset:out[0]+=self.guide*a[0]
        return out
    def diagonal(self):
        d=self.base.diagonal()
        return np.r_[self.factor*self.I+self.guide,d] if self.b.offset else d


class SurfaceMap:
    def __init__(self,sim,d,s):
        self.sim=sim;self.shape=(sim.ns,sim.nd+sim.ns);self.blocks=[];xs=[]
        for i,b in enumerate(sim.bodies):
            dl,dh=sim.doff[i:i+2];sl,sh=sim.soff[i:i+2];q=d[dl:dh]
            R=rotation(b.axis,q[0]) if b.revolute else np.eye(3)
            y=b.X+(b.P@q[b.offset:]+s[sl:sh]).reshape(-1,3)
            x=(y-b.center)@R.T+b.center
            if b.prismatic:x+=q[0]*b.axis
            tangent=(np.cross(b.axis,x-b.center) if b.revolute else np.tile(b.axis,(len(x),1))).ravel() if b.reference else None
            self.blocks.append((b,dl,dh,sl,sh,R,tangent));xs.append(x)
        self.x=np.vstack(xs)
    def __matmul__(self,a):
        out=np.zeros((self.sim.ns,)+a.shape[1:])
        for b,dl,dh,sl,sh,R,tangent in self.blocks:
            v=np.asarray(b.P@a[dl+b.offset:dh])+a[self.sim.nd+sl:self.sim.nd+sh]
            if a.ndim==1:v=(v.reshape(-1,3)@R.T).ravel()
            else:v=np.einsum('ab,nbk->nak',R,v.reshape(-1,3,v.shape[1])).reshape(v.shape)
            if b.reference:v+=tangent.reshape((-1,)+(1,)*(a.ndim-1))*a[dl]
            out[sl:sh]=v
        return out
    def rows(self,world):
        dynamic=[];surface=[]
        from quasistatic_pd import ContactMap
        for b,dl,dh,sl,sh,R,tangent in self.blocks:
            local,material=ContactMap.material_weights(world,np.arange(sl,sh),R)
            modal=sp.csr_matrix(material@b.P)
            if b.reference:modal=sp.hstack((sp.csr_matrix(np.asarray(local@tangent)[:,None]),modal),format='csr')
            dynamic.append(modal);surface.append(material)
        return sp.hstack(dynamic+surface,format='csr')
    def transpose_action(self,force):
        out=np.zeros(self.shape[1])
        for b,dl,dh,sl,sh,R,tangent in self.blocks:
            f=np.asarray(force[sl:sh]);p=(f.reshape(-1,3)@R).ravel()
            ids=np.flatnonzero(p)
            out[dl+b.offset:dh]=np.asarray(b.P[ids].T@p[ids]).ravel()
            if b.reference:out[dl]=tangent@f
            out[self.sim.nd+sl:self.sim.nd+sh]=p
        return out


class SurfaceRows:
    def __init__(self,world,mapping):self.world=world;self.J=mapping;self.shape=(world.shape[0],mapping.shape[1])
    def rows(self,ids):return self.J.rows(self.world.rows(ids))
    def __matmul__(self,a):return self.world@(self.J@a)
    @property
    def T(self):
        from shared_contact import Adjoint
        return Adjoint(self)
    def adjoint(self,w):return self.J.transpose_action(self.world.T@w)


class SurfaceComplianceSystem:
    def __init__(self,sim):self.sim=sim;self.support=None;self.surface_blocks=None
    def set_contact_support(self,world):
        columns=np.unique(np.concatenate([f.cols.ravel() for f in world.faces])) if world.faces else np.empty(0,int)
        self.support=[];self.surface_blocks=[]
        for i,b in enumerate(self.sim.bodies):
            lo,hi=self.sim.soff[i:i+2];ids=columns[(columns>=lo)&(columns<hi)]-lo
            self.support.append(ids)
            self.surface_blocks.append(b.R[np.ix_(ids,ids)] if b.R is not None else None)
        self.sim.statistics['maximum_contact_surface_rows']=max(self.sim.statistics.get('maximum_contact_surface_rows',0),sum(map(len,self.support)))
    def full_surface_response(self,force):
        sim=self.sim;out=np.zeros(sim.ns)
        for i,b in enumerate(sim.bodies):
            lo,hi=sim.soff[i:i+2];p=force[lo:hi];ids=np.flatnonzero(p)
            if b.R is not None and len(ids):out[lo:hi]=b.R[:,ids]@p[ids]
        sim.statistics['surface_recoveries_after_contact_solve']=sim.statistics.get('surface_recoveries_after_contact_solve',0)+1
        return out
    def solve(self,rhs):
        sim=self.sim;out=np.zeros_like(rhs)
        for i,b in enumerate(sim.bodies):
            dl,dh=sim.doff[i:i+2];sl,sh=sim.soff[i:i+2]
            out[dl:dh]=sim.blocks[i].solve(rhs[dl:dh])
            if b.R is not None:
                r=rhs[sim.nd+sl:sim.nd+sh]
                loaded=np.flatnonzero(np.any(r!=0.,axis=1) if r.ndim==2 else r!=0.)
                if self.support is None:
                    if len(loaded):out[sim.nd+sl:sim.nd+sh]=b.R[:,loaded]@r[loaded]
                else:
                    ids=self.support[i]
                    if not np.isin(loaded,ids).all():raise ValueError('Contact load outside certified candidate support')
                    if len(loaded):
                        j=np.searchsorted(ids,loaded)
                        out[sim.nd+sl+ids]=self.surface_blocks[i][:,j]@r[loaded]
        return out


class SurfaceCMSSimulator:
    def __init__(self,scene,bodies,policy=None):
        self.bodies=bodies;self.dt=float(scene['dt']);self.scene={k:v for k,v in scene.items() if k not in ('bodies','material_rules')}
        self.alpha=float((policy or {}).get('hht_alpha',-.05));self.w=1+self.alpha
        self.beta=(1-self.alpha)**2/4;self.gamma=.5-self.alpha
        self.policy=dict(contact_workspace_bytes=256*2**20,contact_qp_tolerance=1e-14,
                         relative_tolerance=1e-6,absolute_tolerance=1e-9,gap_tolerance=1e-9,
                         static_tolerance_m=1e-11,max_iterations=150)|(policy or {})
        self.doff=np.r_[0,np.cumsum([b.rd for b in bodies])];self.nd=int(self.doff[-1])
        self.soff=np.r_[0,np.cumsum([b.X.size for b in bodies])];self.ns=int(self.soff[-1]);self.x_offsets=self.soff//3
        self.d=np.zeros(self.nd);self.v=np.zeros(self.nd);self.acc=np.zeros(self.nd)
        self.s=np.zeros(self.ns);self.p=np.zeros(self.ns);self.old_nonmass=np.zeros(self.nd)
        for i,b in enumerate(bodies):
            if b.reference:
                self.d[self.doff[i]]=b.spec.get('initial_angle' if b.revolute else 'initial_reference_position',0.)
                self.v[self.doff[i]]=b.spec.get('initial_angular_velocity' if b.revolute else 'initial_reference_velocity',0.)
        self.blocks=[DynamicBlock(b,1/(self.beta*self.dt**2*self.w)) for b in bodies]
        radius=float(scene.get('contact_search_distance',.001))
        cache=ContactCandidateCache(.5*radius,32*2**20,'aabb_bvh')
        self.geometry=ConsistentMortarContact.for_scene(bodies,self.x_offsets,scene,radius,
                        candidate_cache=cache,threads=1,gap_rule='overlap_vertices')
        if not scene.get('shared_working_contact'):raise ValueError('Shared lazy contact geometry required')
        self.multiplier_history=ContactMultiplierHistory();self.system=SurfaceComplianceSystem(self)
        self.statistics=dict(online_volume_element_evaluations=0,online_volume_reconstructions=0,
                             online_full_order_factorizations=0,contact_restricted_solves=0)
        self.history=[];self.t=0.;self.x=SurfaceMap(self,self.d,self.s).x
        self.old_nonmass=self.nonmass(self.d,self.v,self.t)

    def external(self,t):
        f=np.zeros(self.nd);ramp=min(1.,t/max(self.dt,self.scene.get('load_ramp',self.dt)))
        for i,b in enumerate(self.bodies):
            if b.reference:f[self.doff[i]]=b.spec.get('torque' if b.revolute else 'reference_force',0.)*ramp
        return f

    def nonmass(self,d,v,t):
        out=-self.external(t)
        for i,b in enumerate(self.bodies):
            lo,hi=self.doff[i:i+2];q=d[lo:hi];speed=v[lo:hi]
            out[lo:hi]+=inertia(b,q,speed)[3]
            out[lo+b.offset:hi]+=np.asarray(b.K@q[b.offset:]).ravel()
            if b.reference:
                k=b.spec.get('guide_stiffness',0.);c=b.spec.get('guide_damping',0.);target=b.spec.get('guide_target_velocity',0.)
                out[lo]+=k*(q[0]-b.spec.get('guide_rest_coordinate',0.)-target*t)+c*(speed[0]-target)
        return out

    def step(self):
        tick=time.perf_counter();h=self.dt;t=self.t+h;old=self.d.copy();oldv=self.v.copy();olda=self.acc.copy()
        predictor=old+h*oldv+h*h*(.5-self.beta)*olda
        velocity_base=oldv+h*(1-self.gamma)*olda
        d=predictor.copy();s=self.s.copy();p=self.p.copy();reference=None;timers=dict(geometry=0.,coupled_solve=0.,surface_mapping=0.,modal=0.)
        for iteration in range(self.policy['max_iterations']):
            start=time.perf_counter();acc=(d-predictor)/(self.beta*h*h);v=velocity_base+self.gamma*h*acc
            nm=self.nonmass(d,v,t);g=(nm-self.alpha/self.w*self.old_nonmass).copy();sc=[]
            for i,b in enumerate(self.bodies):
                lo,hi=self.doff[i:i+2];guide=b.spec.get('guide_stiffness',0.)+b.spec.get('guide_damping',0.)*self.gamma/(self.beta*h)
                block=self.blocks[i];block.update(d[lo:hi],v[lo:hi],guide)
                g[lo:hi]+=block.mass_action(acc[lo:hi])/self.w;sc.extend(block.diagonal())
            scale=np.sqrt(np.maximum(sc,1e-20));timers['modal']+=time.perf_counter()-start
            start=time.perf_counter();mapping=SurfaceMap(self,d,s);x=mapping.x;timers['surface_mapping']+=time.perf_counter()-start
            start=time.perf_counter();gaps,world,_,records=self.geometry.build(x,sp.csr_matrix((x.size,0)))
            G=SurfaceRows(world,mapping);timers['geometry']+=time.perf_counter()-start
            free=np.r_[np.concatenate([-b.solve(g[self.doff[i]:self.doff[i+1]]) for i,b in enumerate(self.blocks)]),-s]
            start=time.perf_counter()
            self.system.set_contact_support(world)
            direction,lam,info=solve_working_qp(self,self.system,G,gaps,free,gaps+G@free,
                self.policy['contact_qp_tolerance'],(records,x))
            target=mapping.transpose_action(world.T@lam);targetp=target[self.nd:]
            targets=self.system.full_surface_response(targetp)
            direction[self.nd:]=targets-s
            residual=float(la.norm((g-target[:self.nd])/scale));static_error=float(np.max(abs(targets-s),initial=0.))
            if reference is None:reference=max(float(la.norm(g/scale)),1e-12)
            threshold=self.policy['absolute_tolerance']+self.policy['relative_tolerance']*reference
            pen=max(0.,-float(gaps.min(initial=0.)));comp=float(np.max(abs(lam*gaps),initial=0.))
            timers['coupled_solve']+=time.perf_counter()-start
            if residual<=threshold and static_error<=self.policy['static_tolerance_m'] and pen<=self.policy['gap_tolerance'] and comp<=1e-8:
                break
            # Safeguarded SQP step for the frozen modal mass/convective force.
            # Only surface vectors are interpolated; no static basis is restored.
            dd=direction[:self.nd];ds=direction[self.nd:];move=np.max(la.norm((mapping@direction).reshape(-1,3),axis=1),initial=0.)
            step=min(1.,.4*self.geometry.radius/max(move,1e-20))
            for i,b in enumerate(self.bodies):
                if b.revolute:step=min(step,.025/max(abs(dd[self.doff[i]]),1e-20))
            penalty=max(1.,1.5*lam.sum());base_static=.5*p@s
            slope=g@dd+p@ds-penalty*pen
            curvature=sum(dd[self.doff[i]:self.doff[i+1]]@block.action(dd[self.doff[i]:self.doff[i+1]]) for i,block in enumerate(self.blocks))
            accepted=False;start=time.perf_counter()
            for backtrack in range(30):
                dt=d+step*dd;st=s+step*ds;pt=p+step*(targetp-p);xt=SurfaceMap(self,dt,st).x
                gt=self.geometry.build_gaps(xt);vp=max(0.,-float(gt.min(initial=0.)))
                change=step*(g@dd)+.5*step*step*curvature+.5*pt@st-base_static+penalty*(vp-pen)
                if change<=1e-4*step*min(slope,0.)+1e-15:
                    d,s,p=dt,st,pt;accepted=True;break
                step*=.5
            timers['geometry']+=time.perf_counter()-start
            if not accepted:raise RuntimeError(f'Surface CMS line search failed: {iteration}, {residual=}, {static_error=}, {pen=}')
        else:raise RuntimeError(f'Surface CMS iteration limit: {residual=}, {static_error=}, {pen=}')
        self.d=d;self.s=s;self.p=p;self.acc=acc;self.v=v;self.t=t;self.x=x
        self.old_nonmass=nm-target[:self.nd]
        self.multiplier_history.commit(records,lam,x)
        nodal=np.asarray(world.T@lam).reshape(-1,3);endpoint=records.endpoint_columns()
        row=dict(t=t,online_s=time.perf_counter()-tick,iterations=iteration+1,residual=residual,threshold=threshold,
                 static_equilibrium_error_m=static_error,complementarity_Nm=comp,
                 pen_um=float(np.max(-endpoint['minimum'],initial=0.))*1e6,
                 force=[nodal[self.x_offsets[i]:self.x_offsets[i+1]].sum(0).tolist() for i in range(len(self.bodies))],
                 refs=[float(d[self.doff[i]]) for i,b in enumerate(self.bodies) if b.reference],timers=timers)
        self.last_nodal_force=nodal;self.history.append(row);return row
