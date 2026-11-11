"""Nodal area-average hard contact, without an independent pointwise guard.

This diagnostic discretization follows the documented distinction between
surface-average constraints and pointwise geometric penetration. It is not an
implementation of Abaqus' proprietary S2S coefficients. The finite overlap,
normal and total gap derivative remain those of our shared geometry backend.
"""
import time
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from consistent_mortar_contact import ConsistentMortarContact
from shared_contact import AllRows, FacePair


class AreaAverageRows(AllRows):
    """Only pressure rows enter equilibrium; vertex gaps remain diagnostics."""
    def __init__(self, faces, size, statistics):
        super().__init__(faces, size, statistics)
        self.blocks = [b for b in self.blocks if b[2] == 'weak']
        self.ends = [b[1] for b in self.blocks]
        self.shape = (self.ends[-1] if self.ends else 0, size)

    def __matmul__(self, direction):
        tick = time.perf_counter()
        values = [f.directional(np.asarray(direction).ravel())[0] for f in self.faces]
        self.stats['directional_checks'] += 1
        self.stats['directional_s'] += time.perf_counter() - tick
        return np.concatenate(values) if values else np.empty(0)


class AreaAverageContact(ConsistentMortarContact):
    recover_invalid_trials = True

    def __init__(self, bodies, offsets, pairs, radius, **options):
        super().__init__(bodies, offsets, pairs, radius, **options)
        # The search envelope is unchanged. There is no 4-micron integration
        # band and no negative-gap exclusion from pressure support.
        # The shared kernel requires inner_radius < search_radius. One ULP
        # below the envelope satisfies its contract without a physical band.
        self.patch = SimpleNamespace(radius=np.nextafter(float(radius), 0.))
        self.retain_negative_pressure_support = True
        self.guard_offset = 0.  # FacePair metadata only; no guard row is solved.
        self.statistics.update(
            common_overlap_tests=0, reused_patch_overlaps=0,
            working_rows_assembled=0, rows_considered=0,
            directional_checks=0, directional_s=0., working_assembly_s=0.,
            pressure_support_mode='nodal_P1_full_search_overlap',
            pointwise_guard_enforced=False, pointwise_limit_m=None,
            integration_search_extent_m=radius,
            coefficient_policy='our_total_gap_derivative_not_native_Abaqus')

    def build(self, x, J, derivatives=True, gaps_only=False):
        tick = time.perf_counter()
        self.statistics['calls'] += 1
        try:
            faces = [FacePair(self, *args, derivatives, gaps_only)
                     for args in self.pair_inputs(x)]
            gaps = (np.concatenate([f.weak_gaps[f.keep] for f in faces])
                    if faces else np.empty(0))
            if gaps_only:
                return gaps
            world = AreaAverageRows(faces, x.size, self.statistics)
            self.statistics['rows_considered'] += len(gaps)
            mapped = (world.mapped(J) if derivatives and J.shape[1]
                      else sp.csr_matrix((len(gaps), J.shape[1])))
            return gaps, world, mapped, world
        finally:
            self.statistics['total_build_seconds'] += time.perf_counter() - tick
