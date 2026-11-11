"""One bounded smooth gap per original secondary face, on unmodified contact triangles."""
import numpy as np
import scipy.sparse as sp
from consistent_mortar_contact import ConsistentMortarContact


def face_blend(gaps, groups, epsilon):
    """Return min over the simplex of a路g + epsilon/2 ||a||虏 and its gradient."""
    unique, inverse=np.unique(groups,return_inverse=True)
    order=np.lexsort((gaps,inverse))
    starts=np.r_[0,np.cumsum(np.bincount(inverse))]
    rows=[];cols=[];values=[];out=[];minimum=[];maximum=[]
    for i in range(len(unique)):
        indices=order[starts[i]:starts[i+1]]
        base=gaps[indices[0]];maximum.append(gaps[indices[-1]]);minimum.append(base)
        # Only vertices within epsilon of the minimum can carry a positive weight.
        z=(gaps[indices]-base)/epsilon
        thresholds=(1+np.cumsum(z))/np.arange(1,len(z)+1)
        active=int(np.count_nonzero(z<thresholds))
        a=thresholds[active-1]-z[:active]
        out.append(base+epsilon*(a@z[:active]+.5*(a@a)))
        rows.extend([i]*active);cols.extend(indices[:active]);values.extend(a)
    W=sp.csr_matrix((values,(rows,cols)),shape=(len(unique),len(gaps)))
    return np.array(out),W,np.array(minimum),np.array(maximum),unique


class BlendedFaceContact(ConsistentMortarContact):
    def __init__(self,*args,blend,**options):
        super().__init__(*args,**options)
        self.blend=float(blend)
        if not np.isfinite(self.blend) or not 0<self.blend<self.radius:
            raise ValueError("Face blend length must be positive and smaller than search radius")

    def face_blending(self,gaps,groups):
        return face_blend(gaps,groups,self.blend)
