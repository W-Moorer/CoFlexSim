"""Linear reference stiffness plus positively weighted sampled PD nonlinear remainder."""
import numpy as np


def linear_fields(body,y):
    t=body.tets;X=body.X
    inv=np.linalg.inv(np.transpose(X[t[:,1:]]-X[t[:,0,None]],(0,2,1)))
    u=y-X
    H=np.transpose(u[t[:,1:]]-u[t[:,0,None]],(0,2,1))@inv
    strain=.5*(H+H.transpose(0,2,1))
    E=body.spec["young"];nu=body.spec["poisson"];mu=E/(2*(1+nu));lam=2*mu*nu/(1-2*nu)
    stress=2*mu*strain+lam*np.trace(strain,axis1=1,axis2=2)[:,None,None]*np.eye(3)
    vol=np.abs(np.linalg.det(X[t[:,1:]]-X[t[:,0,None]]))/6
    G=np.concatenate((-inv.sum(axis=1)[:,None,:],inv),axis=1)
    element_gradient=vol[:,None,None]*(G@stress)
    gradient=np.zeros_like(X);np.add.at(gradient,t,element_gradient)
    F=H+np.eye(3);det=np.linalg.det(F)
    return dict(stress=stress.reshape(-1,9),strain=strain.reshape(-1,9),gradient=gradient,
                energy=float(.5*np.einsum("e,eij,eij",vol,strain,stress)),min_det=float(min(det)),
                max_det=float(max(det)),minimum_singular_values=np.linalg.svd(F,compute_uv=False)[:,-1])

class ReducedMaterial:
    def __init__(self,b,native,rule=None):
        self.body=b;self.groups=[]
        T=b.T
        self.world_base=np.asarray(b.native.K.tocsr()[:,b.free]@T)
        self.reference=T.T@self.world_base[b.free]
        self.reference=(self.reference+self.reference.T)*.5
        self.base=self.reference.copy();self.majorizer=self.reference.copy()
        self.inv_dm=np.linalg.inv(np.transpose(b.X[b.tets[:,1:]]-b.X[b.tets[:,0,None]],(0,2,1)))
        full=np.zeros((b.X.size,T.shape[1]));full[b.free]=T
        tr=full.reshape(len(b.X),3,-1)
        # A conservative bound on each element's material strain-direction norm.
        bounds=[]
        for j in range(0,len(b.tets),48):
            t=b.tets[j:j+48];du=tr[t[:,1:]]-tr[t[:,0,None]]
            df=np.einsum("enak,enc->eack",du,self.inv_dm[j:j+48])
            bounds.append(np.sqrt(np.sum(df*df,axis=(1,2))).max(axis=0))
        self.bounds=np.array(bounds)
        self.selected=0
        if rule is not None:
            ids,weights=np.asarray(rule["ids"]),np.asarray(rule["weights"],float)
            if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or weights.shape!=ids.shape or not len(ids) or len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=len(b.tets)) or not np.all(np.isfinite(weights)&(weights>0)):
                raise ValueError("Invalid material quadrature rule")
            self.selected=len(ids)
            unique=np.unique(weights)
            # For many different weights, integrate one disconnected array of
            # tets. Scaling each reference tetrahedron and its displacements by
            # w**(1/3) preserves F and multiplies its integration volume by w.
            packed=len(unique)>2
            batches=[(1.,ids)] if packed else [(float(w),ids[weights==w]) for w in unique]
            for w,chosen in batches:
                tet=b.tets[chosen]
                if packed:
                    nodes=tet.ravel()
                    indices=np.arange(len(nodes),dtype=np.int32).reshape(tet.shape)
                    scales=np.repeat(np.cbrt(weights),4)
                    X=((b.X[tet]-b.X[tet[:,0,None]])*np.cbrt(weights)[:,None,None]).reshape(-1,3)
                    chain=np.repeat(scales,3)
                else:
                    nodes,indices=np.unique(tet,return_inverse=True)
                    indices=indices.reshape(tet.shape).astype(np.int32)
                    X=b.X[nodes].copy();chain=None
                rows=(3*nodes[:,None]+np.arange(3)).ravel()
                S=np.ascontiguousarray(full[rows])
                if packed:S*=chain[:,None]
                kernel=native.TetModel(X,indices,b.spec["young"],b.spec["poisson"],b.spec["density"])
                KS=kernel.K@S
                Ksub=S.T@KS
                self.base-=w*Ksub
                self.majorizer+=w*(S.T@(kernel.L@S)-Ksub)
                if packed:np.add.at(self.world_base,rows,-chain[:,None]*KS)
                else:self.world_base[rows]-=w*KS
                kernel.release_assembled_operators()
                self.groups.append((w,rows,X,S,kernel,chain))
        self.base=(self.base+self.base.T)*.5;self.majorizer=(self.majorizer+self.majorizer.T)*.5
        self.cached=None;self.evaluations=0;self.element_evaluations=0
        self.full_det_fallbacks=0;self.bound_fallbacks=0;self.bound_passes=0

    def actual_det(self,y):
        t=self.body.tets
        F=np.transpose(y[t[:,1:]]-y[t[:,0,None]],(0,2,1))@self.inv_dm
        minimum=float(np.min(np.linalg.det(F)))
        if minimum<=1e-10:raise RuntimeError("Nonpositive material element determinant")
        return minimum

    def evaluate(self,u,threads):
        if self.cached is not None and np.array_equal(u,self.cached[0]):return self.cached[1]
        g=self.base@u;energy=.5*float(u@g);outputs=[]
        for w,rows,X,S,kernel,chain in self.groups:
            out=kernel.evaluate_parallel(X+(S@u).reshape(-1,3),False,threads)
            f=w*np.asarray(out["gradient"]).ravel()
            g+=S.T@f;energy+=w*out["energy"];outputs.append((rows,f,chain))
        self.evaluations+=1;self.element_evaluations+=self.selected
        eta=float(np.max(self.bounds@abs(u)))
        if eta<.999:minimum=(1.-eta)**3
        else:
            self.full_det_fallbacks+=1
            y=self.body.X.copy();y.ravel()[self.body.free]+=self.body.T@u
            minimum=self.actual_det(y)
        data=dict(energy=energy,reduced_gradient=g,sample_forces=outputs,min_det=minimum)
        self.cached=(u.copy(),data)
        return data

    def world_gradient(self,u,threads):
        a=self.evaluate(u,threads);f=self.world_base@u
        for rows,values,chain in a["sample_forces"]:
            if chain is None:f[rows]+=values
            else:np.add.at(f,rows,chain*values)
        return f.reshape(-1,3)

    def safe_step(self,u,d):
        a=self.bounds@abs(u);b=self.bounds@abs(d)
        if np.all(b<=.8*(1.-a)):
            self.bound_passes+=1;return True
        self.bound_fallbacks+=1
        return False

