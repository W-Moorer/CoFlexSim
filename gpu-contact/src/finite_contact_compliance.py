"""Finite normal enforcement compliance for surface-CMS diagnostics.

The compliance is numerical, not a material layer or clearance. Each pressure
row uses its current overlap area, reference element height, and constrained
normal modulus. Scale 1000 follows the documented augmented-Lagrange order of
magnitude, but the element estimate is ours, not proprietary Abaqus coefficients.
No measured response enters the stiffness. Augmentation is not implemented here;
the candidate is linear penalty and must be identified as such in all results.
"""
import numpy as np
import scipy.sparse as sp
from hard_contact_solver import solve_contact_lcp


class ElementNormalCompliance:
    def __init__(self, specs, scale=1000.):
        if scale <= 0: raise ValueError('Positive enforcement scale required')
        self.scale=float(scale); self.face_k=[]; self.face_h=[]
        for spec in specs:
            X=np.asarray(spec['vertices']); tet=np.asarray(spec['tets']); faces=np.asarray(spec['contact_faces'])
            vertex=X[tet]; volume=np.abs(np.linalg.det((vertex[:,1:]-vertex[:,:1]).transpose(0,2,1)))/6
            owner={}
            for j, indices in enumerate(tet):
                for omit in range(4):
                    key=tuple(sorted(np.delete(indices,omit).tolist()))
                    owner[key]=j if key not in owner else -1
            tri=X[faces]; area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
            ids=np.array([owner[tuple(sorted(f.tolist()))] for f in faces]); assert np.all(ids>=0)
            h=3*volume[ids]/area
            E=float(spec['young']);nu=float(spec['poisson']);normal_modulus=E*(1-nu)/((1+nu)*(1-2*nu))
            self.face_h.append(h);self.face_k.append(self.scale*normal_modulus/h)

    def rows(self, world):
        compliance=[]
        for lo,hi,kind,face in world.blocks:
            if kind!='weak': raise ValueError('Pressure rows required')
            pick=face.wi
            a=face.weak['area'][pick].sum(1)
            # Parallel micro-patches add their normal stiffness, including
            # cross-feature partitions, without duplicating the parent area.
            k=self.face_k[face.slave][face.owners[pick]]
            stiffness=np.bincount(face.inv,a*k,minlength=len(face.area))[face.keep]
            if np.any(stiffness<=0): raise ValueError('Nonpositive contact stiffness')
            compliance.extend(1/stiffness)
        return np.asarray(compliance)


class ElementUniaxialCompliance(ElementNormalCompliance):
    """Alternative E/h enforcement estimate, keeping the declared scale 1000.

    The original uses a laterally constrained modulus. A free contact facet
    does not itself impose zero lateral strain. This opt-in diagnostic uses
    Young's modulus instead; it is not claimed to recover Abaqus' proprietary
    representative-element coefficients and does not implement augmentation.
    No measured pressure, force or trajectory sets this coefficient.
    """
    def __init__(self,specs,scale=1000.):
        super().__init__(specs,scale)
        for i,spec in enumerate(specs):
            nu=float(spec['poisson'])
            self.face_k[i]*=(1+nu)*(1-2*nu)/(1-nu)


def compliant_working_solve(sim,system,G,gaps,free,free_gap,tol,history,compliance):
    """Solve 0 <= lambda _|_ gap + C lambda >= 0 with all-row checking."""
    lam=np.zeros(len(gaps))
    if free_gap.min(initial=0.)>=0: return free,lam,dict(iterations=0,active_count=0)
    seeds=sim.multiplier_history.match(*history,free_gap) if history else {}
    ids=list(seeds);worst=int(np.argmin(free_gap))
    if worst not in ids:ids.append(worst)
    weights=np.array([seeds.get(i,0.) for i in ids]);response=None
    for outer in range(len(gaps)):
        if 8*(3*G.shape[1]*len(ids)+6*len(ids)**2)>sim.policy['contact_workspace_bytes']:
            raise MemoryError('Compliant working-set budget exceeded')
        rows=G.rows(ids if response is None else ids[-1:]);rhs=rows.T.toarray() if sp.issparse(rows) else rows.T
        columns=system.solve(rhs)
        if response is None: response=columns;active=rows
        else:
            response=np.column_stack((response,columns))
            active=sp.vstack((active,rows),format='csr') if sp.issparse(rows) else np.vstack((active,rows))
        small=np.asarray(active@response);small=(small+small.T)*.5
        small.flat[::len(ids)+1]+=compliance[ids]
        weights,info=solve_contact_lcp(small,free_gap[ids],tolerance=tol,
            rank_aware=True,initial=weights,prefer_cholesky=True)
        sim.statistics['contact_restricted_solves']=sim.statistics.get('contact_restricted_solves',0)+1
        direction=free+response@weights;test=gaps+G@direction
        test[ids]+=compliance[ids]*weights;worst_active=float(test[ids].min());test[ids]=np.inf
        j=int(np.argmin(test))
        if test[j]>=-10*tol:
            if worst_active < -100*tol:raise RuntimeError('Compliant contact failed direct feasibility')
            lam[ids]=weights;info.update(stored_contact_columns=len(ids),outer_working_set_iterations=outer+1)
            return direction,lam,info
        ids.append(j);weights=np.r_[weights,0.]
    raise RuntimeError('Unresolved compliant contact working set')


def solve_contact_subproblem(sim,system,G,gaps,free,free_gap,tol,history):
    """Hard behavior is unchanged unless a diagnostic law is explicitly set."""
    law=getattr(sim,'normal_compliance',None)
    if law is None:
        from surface_cms import solve_working_qp
        direction,lam,info=solve_working_qp(sim,system,G,gaps,free,free_gap,tol,history)
        return direction,lam,info,gaps
    compliance=law.rows(history[0])
    direction,lam,info=compliant_working_solve(sim,system,G,gaps,free,free_gap,tol,history,compliance)
    return direction,lam,info,gaps+compliance*lam
