"""Nested, residual-driven static enrichment in the shared PD/HHT loop.

The parent physical-K space is prepared offline. No dynamic modes are added,
no physical model or contact tolerance is changed, and no parent equilibrium
solve is used to choose directions. The reference-energy residual is an
indicator; a uniform nonlinear coercivity constant is not assumed.
"""
import time
import numpy as np
import scipy.linalg as la
from coflex_core import rotation


class AdaptiveCorrection:
    def __init__(self, sim):
        if sim.policy.get('compact_recovery_tolerance'):
            raise ValueError('Adaptive correction currently requires dense prepared basis archives')
        self.sim = sim
        self.relative = float(sim.policy.get('adaptive_relative_tolerance', .02))
        self.absolute = float(sim.policy.get('adaptive_absolute_tolerance', 1e-8))
        self.batch = int(sim.policy.get('adaptive_batch_size', 4))
        if self.relative <= 0 or self.absolute <= 0 or self.batch < 1:
            raise ValueError('Invalid adaptive correction tolerance or batch')
        self.pools = []
        for b in sim.bodies:
            parent = np.asarray(b.T)
            z = parent[:, b.Phi.shape[1]:]
            self.pools.append(dict(Z=z, T=parent,
                L=b.reduced_L.copy(), energy=b.reference_column_energy.copy(),
                trace=b.surface_trace.copy()))
        self.coordinates = [np.eye(b.Z.shape[1]) for b in sim.bodies]
        self.events = []
        self.seconds = 0.
        self.enrichments = 0
        initial = int(sim.policy.get('adaptive_initial_rank', 8))
        if initial < 1: raise ValueError('Adaptive initial rank must be positive')
        ids = [x[:, :min(initial, len(x))] for x in self.coordinates]
        # Initial nonzero states must already lie in the starting subspace.
        self.activate(ids, sim.q.copy(), initialization=True)

    @property
    def selected(self):
        return [np.arange(w.shape[1]) for w in self.coordinates]

    def activate(self, selected, candidate, initialization=False):
        from quasistatic_pd import ReducedBlocks, DiagonalBlock, CondensedSystem
        s = self.sim
        old_offsets = s.q_offsets.copy()
        old_selected = self.coordinates
        old_candidate_x = s.mapping(candidate, jacobian=False)[0].copy()
        old_accepted_x = s.x.copy()
        selected = [np.asarray(w, float) for w in selected]
        offsets = np.r_[0, np.cumsum([b.rd+w.shape[1] for b, w in zip(s.bodies, selected)])]
        transfers=[]
        for b,w,old,pool in zip(s.bodies,selected,old_selected,self.pools):
            energy=pool['energy'][b.Phi.shape[1]:]
            gram=w.T@(energy[:,None]*w)
            transfer=la.solve(gram,w.T@(energy[:,None]*old),assume_a='pos')
            defect=w@transfer-old
            if not initialization and la.norm(np.sqrt(energy)[:,None]*defect)>1e-9*max(1.,la.norm(np.sqrt(energy)[:,None]*old)):
                raise ValueError('Static enrichment must contain the previous space')
            transfers.append(transfer)

        def remap(v):
            if v is None: return None
            out = np.zeros(offsets[-1])
            for i, (b, ids, previous) in enumerate(zip(s.bodies, selected, old_selected)):
                lo, nlo = old_offsets[i], offsets[i]
                out[nlo:nlo+b.rd] = v[lo:lo+b.rd]
                coefficients=transfers[i]@v[lo+b.rd:old_offsets[i+1]]
                if la.norm(ids@coefficients-previous@v[lo+b.rd:old_offsets[i+1]])>1e-11:
                    raise ValueError('State has a nonzero discarded static component')
                out[nlo+b.rd:offsets[i+1]]=coefficients
            return out

        # Validate all transfers before mutating the simulator.
        state = {key: remap(getattr(s, key)) for key in ('q', 'qdot', 'initial_q')}
        qnew = remap(candidate)
        history = {key: remap(getattr(s.time_operator, key))
                   for key in ('old_q', 'old_qdot', 'old_coordinate_acceleration', 'coordinate_acceleration')
                   if hasattr(s.time_operator, key)}
        blocks = []; kblocks = []; dynamic = []; static = []; transforms=[]; energies=[]
        for i, (b, w, pool) in enumerate(zip(s.bodies, selected, self.pools)):
            elastic=la.block_diag(np.eye(b.Phi.shape[1]),w)
            transform=la.block_diag(np.eye(b.offset),elastic)
            blocks.append(transform.T@pool['L']@transform)
            energy=np.sum(elastic*(pool['energy'][:,None]*elastic),axis=0)
            gram=elastic.T@(pool['energy'][:,None]*elastic)
            if la.norm(gram/np.sqrt(np.outer(energy,energy))-np.eye(len(energy)))>1e-8:
                raise ValueError('New static basis must remain K orthogonal')
            kblocks.append(DiagonalBlock(np.r_[np.zeros(b.offset),energy]))
            transforms.append(elastic);energies.append(energy)
            lo = offsets[i]
            dynamic.extend(range(lo, lo+b.rd)); static.extend(range(lo+b.rd, offsets[i+1]))
        L = ReducedBlocks(blocks); K = ReducedBlocks(kblocks)
        groups = [w.shape[1] for w in selected]
        condensed = CondensedSystem(L, dynamic, static, groups, s.policy['static_contact_cache_bytes'])
        tangent = CondensedSystem(K, dynamic, static, groups)
        for system in (condensed, tangent): system.region_cache_limit = int(s.policy['regional_compliance_cache_bytes'])
        for b, w, pool, block, elastic, energy in zip(s.bodies, selected, self.pools, blocks, transforms, energies):
            b.T = np.ascontiguousarray(pool['T']@elastic); b.Z = b.T[:, b.Phi.shape[1]:]
            b.surface_trace = np.ascontiguousarray(pool['trace']@elastic)
            b.reference_column_energy = energy
            b.reduced_L = block; b.r = b.rd+w.shape[1]; b.rotation_cache = {}
            b.projection_cache = None; b.inversion_geometry_cache = []
        s.q_offsets = offsets; s.L = L; s.K = K; s.condensed = condensed; s.tangent_condensed = tangent
        for key, value in state.items(): setattr(s, key, value)
        for key, value in history.items(): setattr(s.time_operator, key, value)
        s._mapping_cache = {}; s._geometry_cache = None
        self.coordinates = [w.copy() for w in selected]
        s.x, s.J, _ = s.mapping(s.q, jacobian='contact')
        actual = s.mapping(qnew, jacobian=False)[0]
        if not np.allclose(actual, old_candidate_x, atol=2e-12, rtol=1e-10) or not np.allclose(s.x, old_accepted_x, atol=2e-12, rtol=1e-10):
            raise RuntimeError('Nested static enrichment changed physical state')
        s.statistics.update(reduced_dimension=len(s.q), algebraic_dimension=len(static), dynamic_dimension=len(dynamic))
        if not initialization: self.enrichments += 1
        return qnew

    def check(self, q, force, Gx, multipliers, projection):
        start = time.perf_counter(); s = self.sim
        _, outputs = s.project(q, gradient=False)
        contact = np.asarray(Gx.T @ multipliers).reshape(-1, 3)
        candidates = []; total2 = 0.; inner2 = 0.; source2 = 0.
        for i, (b, pool, w, f) in enumerate(zip(s.bodies, self.pools, self.coordinates, outputs)):
            sl = slice(s.x_offsets[i], s.x_offsets[i+1]); lo = s.q_offsets[i]
            R = rotation(b.axis, q[lo]) if b.revolute else np.eye(3)
            external = ((force[sl]+contact[sl]) @ R).ravel()[b.free]
            internal = np.asarray(f['gradient']).ravel()[b.free]
            r = pool['Z'].T @ (external-internal)
            src = pool['Z'].T @ external
            energy = pool['energy'][b.Phi.shape[1]:]
            if np.any(energy <= 0): raise ValueError('Static parent metric must be positive')
            gram=w.T@(energy[:,None]*w)
            inner=energy*(w@la.solve(gram,w.T@r,assume_a='pos'))
            remainder=r-inner
            weighted = r*r/energy
            total2 += float(weighted.sum()); inner2 += float(np.sum(inner*inner/energy))
            source2 += float(np.sum(src*src/energy))
            direction=remainder/energy
            for _ in range(2):direction-=w@la.solve(gram,w.T@(energy*direction),assume_a='pos')
            length=np.sqrt(max(float(direction@(energy*direction)),0.))
            if w.shape[1]<w.shape[0] and length>1e-14:
                candidates.append((length*length,i,direction/length))
        iteration = np.sqrt(max(inner2, 0.)); representation = np.sqrt(max(total2-inner2, 0.))
        scale = max(np.sqrt(source2), np.sqrt(max(2*projection[1], 0.)))
        threshold = self.absolute+self.relative*scale
        event = dict(t=s.t+s.dt, static_rank=sum(map(len, self.selected)),
            static_iteration_indicator=iteration, representation_indicator=representation,
            threshold=threshold, metric='physical_reference_K_dual_energy',
            scope='static_parent_representation_indicator_not_trajectory_certificate')
        if representation <= threshold:
            event['action'] = 'accept_representation'
            newq = None
        else:
            selected = [w.copy() for w in self.coordinates]
            candidates.sort(key=lambda item:item[0],reverse=True)
            additions = candidates[:self.batch]
            if not additions: raise RuntimeError('Static bank exhausted above its indicator tolerance')
            for _, i, direction in additions:selected[i]=np.column_stack([selected[i],direction])
            event['action'] = 'enrich_static_space'
            event['added_bodies'] = [i for _, i, _ in additions]
            event['direction']='physical_residual_flexibility_applied_to_unrepresented_balance'
            newq = self.activate(selected, q)
        self.events.append(event); self.seconds += time.perf_counter()-start
        return newq

    def record(self):
        return dict(enrichments=self.enrichments, audit_and_rebuild_seconds=self.seconds,
            active_static_dimensions=[len(x) for x in self.selected],
            parent_static_dimensions=[p['Z'].shape[1] for p in self.pools],
            prepared_parent_storage_bytes=sum(p['T'].nbytes+p['L'].nbytes+p['trace'].nbytes+p['energy'].nbytes for p in self.pools),
            indicator_is_nonlinear_certificate=False, events=self.events)
