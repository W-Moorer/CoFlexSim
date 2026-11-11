"""A periodic interpolatory contact surface attached to the volume mesh by virtual work."""
from types import SimpleNamespace
from functools import lru_cache

import numpy as np
import scipy.sparse as sp

from consistent_mortar_contact import ConsistentMortarContact


def ring_skin(rings, node_count, subdivisions):
    """Cubic, periodic Catmull-Rom in the material ring direction, linear across rings."""
    rings = np.asarray(rings)
    if (rings.ndim != 2 or rings.shape[0] < 2 or rings.shape[1] < 4
            or not np.issubdtype(rings.dtype, np.integer) or rings.min() < 0
            or rings.max() >= node_count or np.unique(rings).size != rings.size):
        raise ValueError("Expected ordered rings of distinct physical surface nodes")
    if not isinstance(subdivisions, int) or subdivisions < 1:
        raise ValueError("Ring subdivision must be a positive integer")
    nz, n = rings.shape
    m = n * subdivisions
    j, remainder = np.divmod(np.arange(m), subdivisions)
    t = remainder / subdivisions
    w = np.column_stack([-.5*t*(1-t)**2, 1-2.5*t*t+1.5*t**3,
                         .5*t+2*t*t-1.5*t**3, -.5*t*t*(1-t)])
    ids = rings[:, (j[:, None] + [-1, 0, 1, 2]) % n]
    shape = (nz*m, node_count)
    lift = sp.csr_matrix((np.tile(w.ravel(), nz),
                          (np.repeat(np.arange(nz*m),4),ids.ravel())), shape=shape)
    lift.eliminate_zeros()
    a = (np.arange(nz-1)[:,None]*m + np.arange(m)).ravel()
    b = (a//m)*m + (a+1)%m
    faces = np.concatenate([np.column_stack([a,b,b+m]), np.column_stack([a,b+m,a+m])])
    return lift, faces.astype(np.int32)


def check_strip_topology(rings, faces):
    """Accept either volume-face diagonal, but preserve the declared trace and winding."""
    def oriented(tri):
        tri=tuple(tri)
        start=tri.index(min(tri))
        return tri[start:]+tri[:start]

    physical={oriented(face) for face in faces}
    n=rings.shape[1]
    if len(physical)!=2*n*(len(rings)-1) or len(physical)!=len(faces):
        raise ValueError("Contact faces do not form the declared two-triangle ring strip")
    for lower,upper in zip(rings[:-1],rings[1:]):
        for i in range(n):
            j=(i+1)%n
            a,b,c,d=lower[i],lower[j],upper[j],upper[i]
            if not ((oriented((a,b,c)) in physical and oriented((a,c,d)) in physical)
                    or (oriented((a,b,d)) in physical and oriented((b,c,d)) in physical)):
                raise ValueError("Periodic strip connectivity or face winding differs from volume mesh")


class EmbeddedMortarContact(ConsistentMortarContact):
    """Use the ordinary finite-overlap kernel on the skin and return physical gradients."""
    def __init__(self, bodies, offsets, pairs, radius, *, skins, **options):
        if options.get("gap_rule") != "overlap_vertices":
            raise ValueError("Embedded hard contact requires all finite-overlap vertex gaps")
        skins = {int(i): spec for i,spec in skins.items()}
        if not set(skins) <= {main for _,main in pairs} or any(i in skins for i,_ in pairs):
            raise ValueError("Declare embedded periodic strips only on main surfaces")
        lifts, proxies, self.physical_supports = [], [], {}
        for i,body in enumerate(bodies):
            faces = body.spec["contact_faces"]
            if i in skins:
                spec = skins[i]
                rings = np.asarray(spec["rings"])
                if not np.array_equal(np.unique(rings), np.unique(faces)):
                    raise ValueError("Embedded rings must cover the declared physical contact trace")
                lift, skin_faces = ring_skin(rings, len(body.X), spec["subdivisions"])
                check_strip_topology(rings,faces)
                faces=skin_faces
            else:
                lift = sp.eye(len(body.X), format="csr")
            lifts.append(lift)
            proxies.append(SimpleNamespace(spec={"contact_faces":faces}))
            self.physical_supports[i] = [np.unique(np.concatenate([
                lift.indices[lift.indptr[k]:lift.indptr[k+1]] for k in face])) + offsets[i] for face in faces]
        self.lift = sp.block_diag(lifts,format="csr")
        self.vector_lift = sp.kron(self.lift,sp.eye(3),format="csr")
        self.skin_offsets = np.r_[0,np.cumsum([p.shape[0] for p in lifts])]
        @lru_cache(maxsize=4096)
        def physical_ids(slave,main,sface,mface):
            side=proxies[slave].spec["contact_faces"][sface]+offsets[slave]
            return np.r_[side,self.physical_supports[main][mface]]
        self.physical_ids=physical_ids
        self.separated_gap={i:float(spec.get("separated_gap",radius)) for i,spec in skins.items()}
        if any(not 0<cut<=radius for cut in self.separated_gap.values()):
            raise ValueError("Positive gap screen must lie within the geometric search radius")
        self.empty = sp.csr_matrix((3*self.lift.shape[0],0))
        super().__init__(proxies,self.skin_offsets,pairs,radius,**options)

    def build(self, x, J, derivatives=True, gaps_only=False):
        result = super().build(self.surface_positions(x),self.empty,derivatives,gaps_only)
        if gaps_only:
            return result
        gaps, derivative, _, records = result
        Gx = (derivative@self.vector_lift).tocsr()
        Gx.eliminate_zeros()
        # Record IDs in physical coordinates. The face index identifies a skin face.
        for row in records:
            slave,main,sface,mface,_ = row["key"]
            row["ids"] = self.physical_ids(slave,main,sface,mface)
        return gaps,Gx,(Gx@J).tocsr(),records

    def surface_positions(self,x):
        return self.lift@x

    def screen_separated_pairs(self,main,owners,mids,secondary,master):
        # A signed distance to a plane is affine over the main triangle. If
        # all three vertices are strictly separated, so is the entire overlap.
        # Re-evaluate at EVERY line-search trial and accepted configuration.
        cut=self.separated_gap.get(main,self.radius)
        if cut==self.radius:
            return owners,mids,secondary,master
        n=np.cross(secondary[:,1]-secondary[:,0],secondary[:,2]-secondary[:,0])
        lengths=np.linalg.norm(n,axis=1)
        distances=np.einsum("nvc,nc->nv",master-secondary[:,:1],n)
        keep=distances.min(axis=1)<=cut*lengths
        return owners[keep],mids[keep],secondary[keep],master[keep]
