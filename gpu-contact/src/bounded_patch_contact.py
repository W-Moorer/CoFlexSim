"""Local weak mortar transmission and a separate pointwise penetration bound."""
import numpy as np
import scipy.sparse as sp
from consistent_mortar_contact import ConsistentMortarContact
from contact_candidate_cache import ContactCandidateCache


class BoundedPatchContact(ConsistentMortarContact):
    def __init__(self,bodies,offsets,pairs,radius,*,bound,reserve=0.,pressure_support='legacy_symmetric',pressure_band=None,**options):
        super().__init__(bodies,offsets,pairs,radius,**options)
        self.bound=float(bound)
        if pressure_support not in ('legacy_symmetric','upper_bounded'):raise ValueError('Unknown pressure support')
        self.pressure_support=pressure_support
        self.retain_negative_pressure_support=pressure_support=='upper_bounded'
        if self.retain_negative_pressure_support and type(self) is BoundedPatchContact:
            raise ValueError('Asymmetric support is implemented in SharedBoundedContact only')
        band=(.5 if self.retain_negative_pressure_support else 2)*self.bound
        if pressure_band is not None:
            if not self.retain_negative_pressure_support or not 0<float(pressure_band)<=self.bound:raise ValueError('Pressure band must be within the local bound')
            band=float(pressure_band)
        if not np.isfinite(band) or not 0<band<radius:
            raise ValueError("Local mortar integration band exceeds the search radius")
        if not np.isfinite(reserve) or not 0<=reserve<self.bound:
            raise ValueError("Invalid vertex guard numerical reserve")
        self.guard_offset=self.bound-reserve
        cache=ContactCandidateCache(band*.5,self.candidate_cache.max_bytes,self.candidate_cache.engine)
        self.patch=ConsistentMortarContact(bodies,offsets,pairs,band,candidate_cache=cache,
                                           threads=self.threads,gap_rule="p1_average")
        self.statistics.update(pressure_support_mode=self.pressure_support,
            pressure_positive_band_m=band,pressure_negative_search_m=radius if self.retain_negative_pressure_support else band,
            vertex_guard_bound_m=self.bound,vertex_guard_offset_m=self.guard_offset)

    def build(self,x,J,derivatives=True,gaps_only=False):
        if gaps_only:
            vertex=super().build(x,None,derivatives=False,gaps_only=True)
            return np.r_[self.patch.build_gaps(x),vertex+self.guard_offset]
        # First block transmits a distributed pressure on the actual near-contact
        # portion. The second checks every original overlap vertex, even when a
        # deeply penetrating region falls outside the local integration band.
        wg,wG,wJ,wr=self.patch.build(x,J,derivatives)
        vg,vG,vJ,vr=super().build(x,J,derivatives)
        G=sp.vstack([wG,vG],format="csr")
        return np.r_[wg,vg+self.guard_offset],G,sp.vstack([wJ,vJ],format="csr"),wr+vr
