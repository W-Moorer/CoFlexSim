"""Global modal projection and residual compliance, before contact restriction.

For H=M/h**2+K and mass-normalized complete modes Phi,
H^-1 = Phi diag(1/(h^-2+omega**2)) Phi.T.
The omitted-mode operator is formed implicitly by subtracting the retained
global modal response; it is not an inverse of a contact-local stiffness block.
"""
import numpy as np
import scipy.linalg as la

class ModalResidualCompliance:
    def __init__(self,retained_modes,bulk_action,bulk_inverse):
        self.V=np.asarray(retained_modes);self.inverse=bulk_inverse
        HV=bulk_action(self.V);self.A=self.V.T@HV;self.A=(self.A+self.A.T)*.5
        self.factor=la.cho_factor(self.A,check_finite=False)

    def modal_response(self,rhs):
        return self.V@la.cho_solve(self.factor,self.V.T@rhs,check_finite=False)

    def residual_response(self,rhs):
        return self.inverse(rhs)-self.modal_response(rhs)

    def contact_compliance(self,contact_jacobian):
        G=np.asarray(contact_jacobian)
        C=G@self.residual_response(G.T)
        return (C+C.T)*.5

    def coupled_direction(self,rhs,W,interface_inverse=None):
        """Solve (H+W W.T)d=rhs in coupled modal/interface coordinates.

        A a + B lambda = V.T rhs
        (I+C_res) lambda - B.T a = W.T F_res rhs
        d = V a + F_res rhs - F_res W lambda

        W includes the PSD contact-curvature square root. The same construction
        with the unweighted normal Jacobian gives the normal residual compliance.
        """
        m=self.V.T@rhs
        residual_free=self.inverse(rhs)-self.V@la.cho_solve(self.factor,m,check_finite=False)
        if W.shape[1]==0:
            a=la.cho_solve(self.factor,m,check_finite=False)
            return self.V@a+residual_free,{'retained_modes':self.V.shape[1],'contact_rank':0,'residual_contact_trace':0.}
        B=self.V.T@W
        full_response=(interface_inverse or self.inverse)(W)
        residual_contact=full_response-self.V@la.cho_solve(self.factor,B,check_finite=False)
        C=W.T@residual_contact;C=(C+C.T)*.5
        factor=la.cho_factor(np.eye(len(C))+C,check_finite=False)
        contact_free=W.T@residual_free
        solve_B=la.cho_solve(factor,B.T,check_finite=False)
        solve_free=la.cho_solve(factor,contact_free,check_finite=False)
        coupled=self.A+B@solve_B
        a=la.solve((coupled+coupled.T)*.5,m-B@solve_free,assume_a='pos',check_finite=False)
        contact_reaction=solve_B@a+solve_free
        d=self.V@a+residual_free-residual_contact@contact_reaction
        return d,{'retained_modes':self.V.shape[1],'contact_rank':W.shape[1],
                  'residual_contact_trace':float(np.trace(C)),
                  'retained_residual_coupling_norm':float(la.norm(B))}
