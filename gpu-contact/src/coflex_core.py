"""One PD/contact backend for all scenes; geometry is input data, not a solver switch.

Native tetrahedral projections + IPC Toolkit 1.4 C++ contact/CCD + sparse/dense
linear algebra. Python assembles the common nonlinear solve and records costs.
The full mode is an in-house discrete consistency reference, not an author baseline.
"""
from pathlib import Path
import sys, time, math
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import ipctk
ipctk.set_num_threads(1)

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BUILD=ROOT/'gpu-contact/cache/extensions'
sys.path.insert(0,str(BUILD/'native/Release'))
import coflex_native

def rotation(axis,theta):
    a=np.asarray(axis,dtype=float);a=a/np.linalg.norm(a)
    C=np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    return np.eye(3)+np.sin(theta)*C+(1-np.cos(theta))*(C@C)

def orthonormal_h(Y,H,existing=None,rank=None):
    if Y.size==0:return np.zeros((H.shape[0],0))
    Y=np.asarray(Y,dtype=float).copy()
    if existing is not None and existing.shape[1]:
        Y-=existing@la.solve(existing.T@(H@existing),existing.T@(H@Y),assume_a='pos')
        Y-=existing@la.solve(existing.T@(H@existing),existing.T@(H@Y),assume_a='pos')
    G=Y.T@(H@Y); vals,vecs=la.eigh((G+G.T)*.5)
    keep=np.flatnonzero(vals>max(1e-18,vals[-1]*1e-11))[::-1]
    if rank is not None:keep=keep[:rank]
    return Y@vecs[:,keep]/np.sqrt(vals[keep])[None,:]

def surface_faces(tets):
    faces=np.concatenate([tets[:,[1,2,3]],tets[:,[0,3,2]],tets[:,[0,1,3]],tets[:,[0,2,1]]])
    keys=np.sort(faces,axis=1);_,first,count=np.unique(keys,axis=0,return_index=True,return_counts=True)
    return faces[first[count==1]].astype(np.int32)

class Body:
    def __init__(self,spec,dt,policy,build_basis=True,native_module=None,assemble_operators=True):
        self.spec=spec;self.X=np.asarray(spec['vertices'],float);self.tets=np.asarray(spec['tets'],np.int32)
        if not assemble_operators and (build_basis or native_module is None):raise ValueError('Geometry-only setup requires an explicit prepared reduced native backend')
        args=(self.X,self.tets,spec['young'],spec['poisson'],spec['density'])
        self.native=(native_module or coflex_native).TetModel(*args) if assemble_operators else native_module.TetModel(*args,False)
        self.mass=np.repeat(self.native.mass,3);self.faces=surface_faces(self.tets)
        self.hub=np.asarray(spec.get('hub_nodes',[]),int);self.fixed=np.asarray(spec.get('fixed_nodes',[]),int)
        self.reference=len(self.hub)>0
        self.revolute=self.reference and spec.get('reference_type','revolute')=='revolute'
        self.prismatic=self.reference and spec.get('reference_type','revolute')=='prismatic'
        if self.reference and not (self.revolute or self.prismatic):raise ValueError('Unknown reference constraint')
        self.center=np.asarray(spec.get('reference_point',[0.,0.,0.]),float)
        self.axis=np.asarray(spec.get('axis',[0.,0.,1.]),float);self.axis/=la.norm(self.axis)
        self.free_nodes=np.setdiff1d(np.arange(len(self.X)),np.union1d(self.hub,self.fixed))
        self.free=(3*self.free_nodes[:,None]+np.arange(3)).reshape(-1)
        self.offset=int(self.reference);self.n=self.offset+len(self.free)
        self.L=self.native.L.tocsc()[self.free,:][:,self.free].tocsc()
        self.K=self.native.K.tocsc()[self.free,:][:,self.free].tocsc()
        self.M=sp.diags(self.mass[self.free],format='csc')
        # The PD frozen-projection majorizer is not physical elastic compliance.
        # CoFlex uses the reference material tangent plus discrete inertia as a
        # metric; exact PD energy/gradient and the original acceptance scale remain.
        self.metric_stiffness=self.K if build_basis else self.L
        self.H=(self.M/dt**2+self.metric_stiffness).tocsc();self.factor=sla.splu(self.H) if build_basis else None
        if build_basis:self.Phi,self.Z,self.selection=self.select_basis(policy)
        else:
            self.Phi=np.zeros((len(self.free),0));self.Z=self.Phi.copy()
            self.selection={'criterion':'not_used_by_full_order_reference','principal_dimension':0,'flexibility_dimension':0}
        self.q=np.zeros(self.n)
        if self.reference:self.q[0]=spec.get('initial_angle',0.) if self.revolute else spec.get('initial_reference_position',0.)
        if 'initial_displacement' in spec:self.q[self.offset:]=np.asarray(spec['initial_displacement']).reshape(-1)[self.free]
        self.enrichments=0
        self.interface_response_cache={}
        self.interface_cache_limit=policy.get('interface_response_cache_columns',96)
        self.interface_cache_misses=0
        node_rows=np.broadcast_to(3*self.free_nodes[:,None,None]+np.arange(3)[None,None,:],(len(self.free_nodes),3,3)).reshape(-1)
        nr=3*len(self.X)
        self.j_indices=np.r_[np.arange(nr),node_rows].astype(np.int32) if self.reference else node_rows.astype(np.int32)
        self.j_indptr=np.r_[0,nr+3*np.arange(len(self.free)+1)].astype(np.int32) if self.reference else (3*np.arange(len(self.free)+1)).astype(np.int32)

    def load_channels(self):
        # Geometry-derived spatial patches and all three force directions.
        xyz=self.X[self.free_nodes];origin=xyz.mean(axis=0);extent=np.maximum(np.ptp(xyz,axis=0),1e-12)
        coords=(xyz-origin)/extent
        patch=(coords[:,0]>0).astype(int)+2*(coords[:,1]>0)+4*(coords[:,2]>0)
        cols=[]
        if self.revolute:
            radial=self.X[self.free_nodes]-self.center
            for response in [np.cross(self.axis,radial),np.cross(self.axis,np.cross(self.axis,radial))]:
                column=(self.native.mass[self.free_nodes,None]*response).reshape(-1)
                if la.norm(column)>0:cols.append(column/la.norm(column))
        for j in range(8):
            nodes=np.flatnonzero(patch==j)
            for d in range(3):
                b=np.zeros(len(self.free));b[3*nodes+d]=1/max(1,np.sqrt(len(nodes)));cols.append(b)
        # Local patches around distributed surface landmarks also excite tooth bending.
        surface=np.intersect1d(np.unique(self.faces),self.free_nodes)
        if len(surface):
            angles=np.arctan2(self.X[surface,1]-self.center[1],self.X[surface,0]-self.center[0])
            landmarks=surface[np.argsort(angles)[np.linspace(0,len(surface)-1,16).astype(int)]]
            for node in landmarks:
                distance=la.norm(xyz-self.X[node],axis=1);ids=np.argsort(distance)[:min(5,len(xyz))]
                for d in range(3):
                    b=np.zeros(len(self.free));b[3*ids+d]=1/np.sqrt(len(ids));cols.append(b)
        return np.stack(cols,axis=1)

    def select_basis(self,policy):
        n=len(self.free);B=self.load_channels();pool=min(policy['candidate_modes'],n-1)
        if n<=220:
            vals,vecs=la.eigh(self.K.toarray(),self.M.toarray());vals,vecs=vals[:pool],vecs[:,:pool]
        else:
            vals,vecs=sla.eigsh(self.K,k=pool,M=self.M,sigma=-1e-3,which='LM',tol=1e-8,v0=np.linspace(1.,2.,n))
            order=np.argsort(vals);vals,vecs=vals[order],vecs[:,order]
        vals=np.maximum(vals,0);freq=np.sqrt(vals)/(2*np.pi)
        bands=np.asarray(self.spec.get('frequency_band_hz',[0.,50.,100.,200.]))
        # Energy-response weighting and normalized load participation. No fictitious damping.
        participation=la.norm(vecs.T@B,axis=1)
        denominator=np.min(np.abs(vals[:,None]-(2*np.pi*bands[None,:])**2),axis=1)
        score=participation*np.sqrt(np.maximum(vals,1.))/(denominator+1e-12*max(1.,vals.max()))
        order=np.argsort(score)[::-1];selected=set(order[:min(policy['principal_modes'],pool)].tolist())
        # Free rigid motions and near-degenerate eigenvalue clusters cannot be split.
        selected.update(np.flatnonzero(vals<max(1e-7,vals.max()*1e-10)).tolist())
        for i in list(selected):selected.update(np.flatnonzero(np.abs(vals-vals[i])<=1e-5*max(1.,vals[i])).tolist())
        selected=np.array(sorted(selected),int)
        Phi=orthonormal_h(vecs[:,selected],self.H)
        Y=self.factor.solve(B);Z=orthonormal_h(Y,self.H,Phi,policy['flexibility_modes'])
        record=dict(candidate_frequencies_hz=freq.tolist(),scores=score.tolist(),selected_indices=selected.tolist(),
                    principal_dimension=Phi.shape[1],flexibility_dimension=Z.shape[1],
                    criterion='load_participation_energy_response_over_declared_band_v1',
                    candidate_completeness_certified=False)
        return Phi,Z,record

    def local(self,q):
        x=self.X.copy();x.reshape(-1)[self.free]+=q[self.offset:];return x

    def positions(self,q):
        local=self.local(q);R=rotation(self.axis,q[0]) if self.revolute else np.eye(3)
        world=(local-self.center)@R.T+self.center
        if self.prismatic:world+=q[0]*self.axis
        return world,local,R

    def mapping(self,q):
        world,local,R=self.positions(q)
        values=np.tile(R.T.reshape(-1),len(self.free_nodes))
        if self.reference:
            tangent=(np.cross(self.axis,world-self.center) if self.revolute else np.tile(self.axis,(len(world),1))).reshape(-1)
            values=np.r_[tangent,values]
        J=sp.csc_matrix((values,self.j_indices,self.j_indptr),shape=(3*len(self.X),self.n),copy=False)
        return world,J,local

    def basis(self):
        V=sp.block_diag(([np.ones((1,1))] if self.reference else [])+[np.c_[self.Phi,self.Z]],format='csc')
        return V,self.offset+self.Phi.shape[1]

    def enrich(self,g,limit):
        if self.Phi.shape[1]+self.Z.shape[1]>=min(len(self.free),limit):return False
        z=orthonormal_h(self.factor.solve(g[self.offset:])[:,None],self.H,np.c_[self.Phi,self.Z])
        if not z.shape[1]:return False
        self.Z=np.c_[self.Z,z];self.enrichments+=1;return True

    def curve_envelope(self,q,p,alpha):
        if not self.revolute:return 0.
        dtheta=alpha*p[0];u0=self.local(q)-self.center
        du=np.zeros_like(u0);du.reshape(-1)[self.free]=alpha*p[self.offset:]
        radius=max(la.norm(u0,axis=1).max(),la.norm(u0+du,axis=1).max())
        return (dtheta*dtheta*radius+2*abs(dtheta)*la.norm(du,axis=1).max())/8.

    def compliance_action(self,rhs,sparse_interface=False):
        if not sparse_interface or rhs.ndim==1:return self.factor.solve(np.asfortranarray(rhs))
        ids=np.flatnonzero(np.any(rhs!=0.,axis=1))
        if not len(ids):return np.zeros_like(rhs)
        if len(ids)>self.interface_cache_limit:return self.factor.solve(np.asfortranarray(rhs))
        missing=[int(j) for j in ids if int(j) not in self.interface_response_cache]
        # Preserve useful exact columns when a moving interface adds a few DOFs.
        # Python dict order is used as an LRU queue; no cached value is modified.
        requested=set(map(int,ids))
        for j in ids:
            j=int(j)
            if j in self.interface_response_cache:
                value=self.interface_response_cache.pop(j);self.interface_response_cache[j]=value
        while len(self.interface_response_cache)+len(missing)>self.interface_cache_limit:
            victim=next(j for j in self.interface_response_cache if j not in requested)
            del self.interface_response_cache[victim]
        if missing:
            unit=np.zeros((len(self.free),len(missing)),order='F');unit[missing,np.arange(len(missing))]=1.
            responses=self.factor.solve(unit);self.interface_cache_misses+=len(missing)
            for k,j in enumerate(missing):self.interface_response_cache[j]=responses[:,k].copy()
        response=np.column_stack([self.interface_response_cache[int(j)] for j in ids])
        return response@rhs[ids]

DEFAULT_POLICY=dict(principal_modes=12,flexibility_modes=20,candidate_modes=48,
                    maximum_basis=100,max_iterations=250,relative_tolerance=1e-5,absolute_tolerance=1e-8,
                    acceleration_history=12,linear_defect_forcing=.2,interface_response_cache_columns=96)

class Simulator:
    def __init__(self,scene,mode='coflex_pd',policy=None):
        if mode not in ['coflex_pd','full_order']:raise ValueError('Unknown common-core mode')
        start=time.perf_counter();self.scene=scene;self.mode=mode;self.dt=scene['dt'];self.policy=DEFAULT_POLICY|dict(policy or {})
        self.bodies=[Body(b,self.dt,self.policy,mode=='coflex_pd') for b in scene['bodies']]
        self.q_offsets=np.r_[0,np.cumsum([b.n for b in self.bodies])]
        self.x_offsets=np.r_[0,np.cumsum([len(b.X) for b in self.bodies])]
        self.q=np.concatenate([b.q for b in self.bodies]);self.mass=np.concatenate([b.mass for b in self.bodies])
        self.elastic_H=sp.block_diag([sp.block_diag(([sp.csc_matrix((1,1))] if b.reference else [])+[b.metric_stiffness],format='csc') for b in self.bodies],format='csc')
        self._acceptance_diagonal_shift=np.concatenate([np.r_[([0.] if b.reference else []),(b.L-b.metric_stiffness).diagonal()] for b in self.bodies])
        self.x,self.J,_=self.mapping(self.q)
        self.faces=np.concatenate([b.faces+self.x_offsets[i] for i,b in enumerate(self.bodies)]).astype(np.int32)
        self.mesh=ipctk.CollisionMesh.build_from_full_mesh(self.x,ipctk.edges(self.faces),self.faces)
        self.dhat=scene['contact_distance'];self.kappa=scene['barrier_stiffness'];self.barrier=ipctk.BarrierPotential(self.dhat)
        self.v=np.zeros_like(self.x)
        for i,b in enumerate(self.bodies):
            sl=slice(self.x_offsets[i],self.x_offsets[i+1]);self.v[sl]=b.spec.get('initial_velocity',[0.,0.,0.])
            if b.revolute:self.v[sl]+=b.spec.get('initial_angular_velocity',0.)*np.cross(b.axis,self.x[sl]-b.center)
            if b.prismatic:self.v[sl]+=b.spec.get('initial_reference_velocity',0.)*b.axis
        if ipctk.has_intersections(self.mesh,self.mesh.vertices(self.x)):raise ValueError('Intersecting initial geometry')
        self.t=0.;self.history=[];self.preprocess_seconds=time.perf_counter()-start
        self.timers={k:0. for k in ['local_projection','contact','linear_solve','basis_update','ccd','mapping','line_search']}
        self.qdot=np.zeros_like(self.q)
        for i,b in enumerate(self.bodies):
            if b.revolute:self.qdot[self.q_offsets[i]]=b.spec.get('initial_angular_velocity',0.)
            elif b.prismatic:self.qdot[self.q_offsets[i]]=b.spec.get('initial_reference_velocity',0.)
            else:self.qdot[self.q_offsets[i]:self.q_offsets[i+1]]=self.v[self.x_offsets[i]:self.x_offsets[i+1]].reshape(-1)[b.free]
        self._basis_key=None
        self.condensed_correction_calls=0

    def mapping(self,q):
        states=[b.mapping(q[self.q_offsets[i]:self.q_offsets[i+1]]) for i,b in enumerate(self.bodies)]
        if not hasattr(self,'_j_indices'):
            self._j_indices=np.concatenate([s[1].indices+3*self.x_offsets[i] for i,s in enumerate(states)]).astype(np.int32)
            counts=np.r_[0,np.cumsum([s[1].nnz for s in states])]
            self._j_indptr=np.r_[np.concatenate([s[1].indptr[:-1]+counts[i] for i,s in enumerate(states)]),counts[-1]].astype(np.int32)
        J=sp.csc_matrix((np.concatenate([s[1].data for s in states]),self._j_indices,self._j_indptr),shape=(3*self.x_offsets[-1],len(q)),copy=False)
        return np.vstack([s[0] for s in states]),J,[s[2] for s in states]

    def positions(self,q):
        states=[b.positions(q[self.q_offsets[i]:self.q_offsets[i+1]]) for i,b in enumerate(self.bodies)]
        return np.vstack([s[0] for s in states]),[s[1] for s in states]

    def assemble(self,q,pred,torques,derivatives=True):
        tick=time.perf_counter()
        if derivatives:x,J,local=self.mapping(q)
        else:x,local=self.positions(q);J=None
        self.timers['mapping']+=time.perf_counter()-tick
        delta=x.reshape(-1)-pred.reshape(-1);energy=.5*np.dot(self.mass*delta,delta)/self.dt**2
        g=np.asarray(J.T@(self.mass*delta/self.dt**2)).reshape(-1) if derivatives else np.zeros(len(q));ge=np.zeros(len(q))
        tick=time.perf_counter();mindet=1e10
        for i,b in enumerate(self.bodies):
            e=b.native.evaluate(local[i]);energy+=e['energy'];mindet=min(mindet,e['min_det'])
            begin=self.q_offsets[i];ge[begin+b.offset:self.q_offsets[i+1]]=np.asarray(e['gradient']).reshape(-1)[b.free]
            if b.reference:energy-=torques[i]*q[begin];ge[begin]-=torques[i]
        g+=ge;self.timers['local_projection']+=time.perf_counter()-tick
        tick=time.perf_counter();surface=self.mesh.vertices(x);collisions=ipctk.NormalCollisions();collisions.build(self.mesh,surface,self.dhat)
        energy+=self.kappa*self.barrier(collisions,self.mesh,surface)
        contact_g=self.kappa*self.mesh.to_full_dof(self.barrier.gradient(collisions,self.mesh,surface)) if derivatives else None
        if derivatives:g+=np.asarray(J.T@contact_g).reshape(-1)
        H=None
        if derivatives:
            CH=self.kappa*self.mesh.to_full_dof(self.barrier.hessian(collisions,self.mesh,surface,ipctk.PSDProjectionMethod.CLAMP))
            H=(self.elastic_H+J.T@(sp.diags(self.mass/self.dt**2,format='csc')+CH)@J).tocsc()
            self._acceptance_scale=np.sqrt(np.maximum(H.diagonal()+self._acceptance_diagonal_shift,1e-20))
            self._contact_H=CH;self._linearization_q=q.copy();self._linearization_J=J
            self._bulk_reference_cache={}
        self.timers['contact']+=time.perf_counter()-tick
        return energy,g,H,x,J,mindet,contact_g,len(collisions)

    def basis(self):
        key=tuple((b.Phi.shape[1],b.Z.shape[1],b.enrichments) for b in self.bodies)
        if key==self._basis_key:return self._basis_V,self._basis_r
        blocks=[];retained=[];correction=[];offset=0
        bulk=[];self._body_basis=[];self._body_basis_offsets=[0]
        for b in self.bodies:
            V,r=b.basis();blocks.append(V);retained.extend(range(offset,offset+r));correction.extend(range(offset+r,offset+V.shape[1]));offset+=V.shape[1]
            U=np.c_[b.Phi,b.Z];self._body_basis.append(U)
            bulk.append(la.block_diag(*(([np.zeros((1,1))] if b.reference else [])+[U.T@(b.H@U)])))
            self._body_basis_offsets.append(offset)
        order=retained+correction
        self._basis_key=key;self._basis_order=np.array(order);self._bulk_A=la.block_diag(*bulk)
        self._basis_V=sp.block_diag(blocks,format='csc')[:,order];self._basis_r=len(retained)
        return self._basis_V,self._basis_r

    def projected_hessian(self,V):
        A=self._bulk_A.copy()
        for i,b in enumerate(self.bodies):
            if not b.reference:continue
            q=self._linearization_q[self.q_offsets[i]:self.q_offsets[i+1]]
            tangent=np.cross(b.axis,b.local(q)-b.center) if b.revolute else np.tile(b.axis,(len(b.X),1))
            a=self._body_basis_offsets[i];end=self._body_basis_offsets[i+1]
            A[a,a]=np.sum(b.mass*tangent.reshape(-1)**2)/self.dt**2
            coupling=(b.mass*tangent.reshape(-1))[b.free]@self._body_basis[i]/self.dt**2
            A[a,a+1:end]=coupling;A[a+1:end,a]=coupling
        A=A[np.ix_(self._basis_order,self._basis_order)]
        ids=np.unique(self._contact_H.nonzero()[0])
        if len(ids):
            W=(self._linearization_J[ids,:]@V).toarray()
            A+=W.T@(self._contact_H.tocsr()[ids,:][:,ids]@W)
        return (A+A.T)*.5

    def direction(self,H,g):
        if self.mode=='full_order':return sla.spsolve(H,-g),0,False
        V,r=self.basis();A=self.projected_hessian(V);rhs=-np.asarray(V.T@g).reshape(-1)
        try:
            if r<A.shape[0]:
                zz=la.cho_factor(A[r:,r:],check_finite=False)
                zq=la.cho_solve(zz,A[r:,:r],check_finite=False);zb=la.cho_solve(zz,rhs[r:],check_finite=False)
                S=A[:r,:r]-A[:r,r:]@zq
                q=la.solve((S+S.T)*.5,rhs[:r]-A[:r,r:]@zb,assume_a='pos',check_finite=False)
                z=zb-zq@q;d=V@np.r_[q,z]
            else:d=V@la.solve(A,rhs,assume_a='pos',check_finite=False)
        except la.LinAlgError:
            return sla.spsolve(H,-g),V.shape[1],True
        d=np.asarray(d).reshape(-1)
        defect=-g-H@d;scale=np.sqrt(np.maximum(H.diagonal(),1e-20))
        if la.norm(defect/scale)>self.policy['linear_defect_forcing']*max(la.norm(g/scale),1e-30):
            d+=self.condensed_residual_correction(H,defect)
            self.condensed_correction_calls+=1
        return d,V.shape[1],False

    def bulk_inverse(self,rhs,sparse_interface=False):
        """Apply the reference elastic-tangent + inertia compliance.

        Rotating reference points are eliminated by a scalar Schur complement.
        No factorization of the changing full coupled matrix is performed here.
        This is a solver metric for the PD incremental potential, not its
        frozen-projection majorizer or the exact deformed material tangent.
        """
        vector=rhs.ndim==1;rhs=np.asarray(rhs).reshape(len(self.q),-1);out=np.empty_like(rhs)
        for i,b in enumerate(self.bodies):
            lo,hi=self.q_offsets[i:i+2];r=rhs[lo:hi];u=b.compliance_action(r[b.offset:],sparse_interface)
            if b.reference:
                if i not in self._bulk_reference_cache:
                    q=self._linearization_q[lo:hi]
                    tangent=(np.cross(b.axis,b.local(q)-b.center) if b.revolute else np.tile(b.axis,(len(b.X),1))).reshape(-1)
                    c=(b.mass*tangent)[b.free]/self.dt**2;a=np.dot(b.mass,tangent*tangent)/self.dt**2
                    y=b.factor.solve(c);S=a-np.dot(c,y)
                    self._bulk_reference_cache[i]=(c,y,S)
                c,y,S=self._bulk_reference_cache[i]
                if S<=0:raise RuntimeError('Nonpositive reference-point Schur complement')
                theta=(r[0]-c@u)/S;out[lo]=theta;out[lo+1:hi]=u-y[:,None]*theta[None,:]
            else:out[lo:hi]=u
        return out[:,0] if vector else out

    def condensed_residual_correction(self,H,rhs):
        """Exact bulk compliance plus a current-contact Woodbury closure.

        This corrects the unresolved physical residual after a reduced solve;
        the original author solver is never called or modified.
        """
        ids=np.unique(self._contact_H.nonzero()[0]);base=self.bulk_inverse(rhs)
        if not len(ids):return base
        if len(ids)>800:
            P=sla.LinearOperator(H.shape,matvec=self.bulk_inverse)
            correction,info=sla.cg(H,rhs,M=P,rtol=1e-9,atol=1e-14,maxiter=400)
            if info:raise RuntimeError('Condensed residual PCG did not converge')
            return correction
        C=self._contact_H.tocsr()[ids,:][:,ids].toarray()
        vals,U=la.eigh((C+C.T)*.5);keep=vals>max(1e-15,vals[-1]*1e-13)
        W=np.asarray(self._linearization_J[ids,:].T@(U[:,keep]*np.sqrt(vals[keep])[None,:]))
        if not W.shape[1]:return base
        CW=self.bulk_inverse(W,sparse_interface=True);S=np.eye(W.shape[1])+W.T@CW
        return base-CW@la.solve((S+S.T)*.5,W.T@base,assume_a='pos',check_finite=False)

    def safe_path(self,q,p,alpha,x):
        tick=time.perf_counter();target,_=self.positions(q+alpha*p)
        epsilon=max(b.curve_envelope(q[self.q_offsets[i]:self.q_offsets[i+1]],p[self.q_offsets[i]:self.q_offsets[i+1]],alpha) for i,b in enumerate(self.bodies))
        safe=ipctk.is_step_collision_free(self.mesh,self.mesh.vertices(x),self.mesh.vertices(target),2*epsilon+1e-12)
        self.timers['ccd']+=time.perf_counter()-tick;return bool(safe)

    def step(self):
        begin=time.perf_counter();dt=self.dt;q=self.q.copy();oldx=self.x.copy();t=self.t+dt
        f=np.zeros_like(self.x);torques=[]
        for i,b in enumerate(self.bodies):
            ramp=min(1.,t/max(dt,self.scene.get('load_ramp',dt)))
            sl=slice(self.x_offsets[i],self.x_offsets[i+1]);f[sl]=np.asarray(b.spec.get('body_acceleration',[0.,0.,0.]))*b.native.mass[:,None]
            f[sl]+=np.asarray(b.spec.get('nodal_force',np.zeros_like(b.X)))*ramp
            torques.append(b.spec.get('torque' if b.revolute else 'reference_force',0.)*ramp)
        pred=oldx+dt*self.v+dt*dt*f/self.mass.reshape(-1,3)
        initial=self.assemble(q,pred,torques)
        initial_reference=max(la.norm(initial[1]/self._acceptance_scale),1e-12)
        predictor=dt*self.qdot
        if la.norm(predictor)>0 and self.safe_path(q,predictor,1.,oldx):
            try:
                if self.assemble(q+predictor,pred,torques,False)[0]<self.assemble(q,pred,torques,False)[0]:q+=predictor
            except RuntimeError:pass
        reference=initial_reference;trace=[];fallbacks=0;enriched=0;basis_dim=0;condensed_corrections=0
        correction_count_start=self.condensed_correction_calls
        pairs=[];previous_q=None;previous_g=None
        for iteration in range(self.policy['max_iterations']):
            E,g,H,x,J,mindet,cg,ncontacts=self.assemble(q,pred,torques)
            residual=la.norm(g/self._acceptance_scale)
            if reference is None:reference=max(residual,1e-12)
            threshold=self.policy['absolute_tolerance']+self.policy['relative_tolerance']*reference
            if residual<=threshold:break
            if previous_q is not None:
                s=q-previous_q;y=g-previous_g;sy=np.dot(s,y)
                if sy>1e-10*la.norm(s)*la.norm(y) and sy>1e-30:
                    pairs.append((s,y,1/sy));pairs=pairs[-self.policy['acceleration_history']:]
            # The same safeguarded L-BFGS acceleration surrounds the PD inverse
            # in both in-house modes. CoFlex supplies its condensed inverse action.
            loop=g.copy();coeff=[]
            for s,y,rho in reversed(pairs):
                a=rho*np.dot(s,loop);coeff.append(a);loop-=a*y
            tick=time.perf_counter();p,basis_dim,fb=self.direction(H,loop);fallbacks+=int(fb)
            inverse=-p
            for (s,y,rho),a in zip(pairs,reversed(coeff)):inverse+=s*(a-rho*np.dot(y,inverse))
            p=-inverse
            if np.dot(g,p)>=0:
                pairs=[];p,basis_dim,fb=self.direction(H,g);fallbacks+=int(fb)
            self.timers['linear_solve']+=time.perf_counter()-tick
            # If Galerkin correction stalls while the physical residual is large,
            # add the H-complement residual response; never silently accept ROM stationarity.
            gain=-np.dot(g,p)
            if self.mode=='coflex_pd' and gain<.08*residual**2:
                tick=time.perf_counter();changed=False
                for i,b in enumerate(self.bodies):changed|=b.enrich(g[self.q_offsets[i]:self.q_offsets[i+1]],self.policy['maximum_basis'])
                self.timers['basis_update']+=time.perf_counter()-tick
                if changed:enriched+=1;previous_q=None;previous_g=None;continue
                tick=time.perf_counter()
                # Residual correction includes current contact stiffness and the
                # reference-point coupling, not just the elastic right-hand side.
                unresolved=-g-H@p
                p+=self.condensed_residual_correction(H,unresolved);condensed_corrections+=1
                self.timers['linear_solve']+=time.perf_counter()-tick
                gain=-np.dot(g,p)
            if not np.isfinite(gain) or gain<=0:raise RuntimeError('No descent direction')
            previous_q=q.copy();previous_g=g.copy()
            alpha=1.
            for i,b in enumerate(self.bodies):
                qs=q[self.q_offsets[i]:self.q_offsets[i+1]];ps=p[self.q_offsets[i]:self.q_offsets[i+1]]
                dx=np.zeros_like(b.X);dx.reshape(-1)[b.free]=ps[b.offset:]
                alpha=min(alpha,b.native.inversion_step(b.local(qs),dx))
            tick=time.perf_counter();accepted=False
            for ls in range(35):
                if self.safe_path(q,p,alpha,x):
                    En=self.assemble(q+alpha*p,pred,torques,False)[0]
                    if np.isfinite(En) and En<=E-1e-4*alpha*gain+1e-14:
                        q+=alpha*p;accepted=True;trace.append([float(E),float(En),float(alpha),float(residual)]);break
                alpha*=.5
            self.timers['line_search']+=time.perf_counter()-tick
            if not accepted:raise RuntimeError(f'Line search failed at step {len(self.history)} residual {residual:g}')
        else:raise RuntimeError(f'PD did not converge: residual {residual:g}, target {threshold:g}')
        self.qdot=(q-self.q)/dt;self.q=q;self.x=x;self.J=J;self.v=(self.x-oldx)/dt;self.t=t
        hub_error=0.;angles=[];reference_coordinates=[];contact_resultants=[];contact_moments=[]
        for i,b in enumerate(self.bodies):
            sl=slice(self.x_offsets[i],self.x_offsets[i+1]);forces=-cg.reshape(-1,3)[sl]
            contact_resultants.append(forces.sum(axis=0).tolist())
            contact_moments.append(np.cross(self.x[sl]-b.center,forces).sum(axis=0).tolist())
            if b.reference:
                theta=q[self.q_offsets[i]];reference_coordinates.append(float(theta))
                if b.revolute:angles.append(float(theta))
                exact=((b.X[b.hub]-b.center)@rotation(b.axis,theta).T+b.center) if b.revolute else b.X[b.hub]+theta*b.axis
                hub_error=max(hub_error,float(np.max(np.abs(self.x[self.x_offsets[i]+b.hub]-exact))))
        row=dict(t=t,seconds=time.perf_counter()-begin,iterations=iteration+1,residual=residual,threshold=threshold,
                 min_det=float(mindet),contacts=ncontacts,hub_error=hub_error,angles=angles,reference_coordinates=reference_coordinates,
                 contact_resultants=contact_resultants,contact_moments=contact_moments,
                 contact_force_norm=float(la.norm(cg)),basis_dimension=basis_dim,enrichment_rounds=enriched,full_order_fallbacks=fallbacks,
                 condensed_residual_corrections=condensed_corrections+self.condensed_correction_calls-correction_count_start,
                 interface_response_cache_columns=sum(len(b.interface_response_cache) for b in self.bodies),
                 interface_response_cache_misses=sum(b.interface_cache_misses for b in self.bodies),
                 accepted_energy_decreases=all(a[1]<=a[0]+1e-12 for a in trace))
        self.history.append(row);return row

    def fields(self):
        result=[]
        for i,b in enumerate(self.bodies):
            q=self.q[self.q_offsets[i]:self.q_offsets[i+1]]
            field=b.native.evaluate(b.local(q),True)
            if b.revolute:
                R=rotation(b.axis,q[0]);field['stress']=np.einsum('ij,njk,lk->nil',R,np.asarray(field['stress']).reshape(-1,3,3),R).reshape(-1,9)
            result.append(field)
        return result
