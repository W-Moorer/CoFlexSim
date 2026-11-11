"""Explicit numerical contract shared by the region-storage CPU/CUDA paths.

Backend selection changes storage/execution only. Unsupported contact laws are
rejected, rather than silently falling back to the historical pressure law.
"""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ContactContract:
    pressure: str = 'smoothed_nodal_median'
    gap: str = 'p1_average'
    enforcement: str = 'hard_complementarity'
    force_time: str = 'physical_endpoint'
    friction: str = 'none'
    alpha: float = -.05

    def validate(self,scene):
        if self != ContactContract():
            raise NotImplementedError('This common CPU/CUDA release supports the declared original median hard-contact law only')
        if scene.get('contact_force_convention') != self.force_time:
            raise ValueError('Contact-force time convention differs from shared contract')
        if float(scene['dt']) <= 0:
            raise ValueError('Positive physical time step required')


def create_backend(scene,bodies,backend,policy=None,batch_capacity=64,sources=None):
    if backend in ('native_cuda','host_cuda'):
        import copy
        if not sources or any(not getattr(s,'sealed',False) for s in sources):
            raise ValueError('Optimized CUDA execution requires a sealed residual library; no hidden online FE solve')
        sc=copy.deepcopy(scene)
        sc.update(regional_incremental_updates=True,regional_compiled_query=True,
            regional_parallel_assembly=True,regional_device_candidates=True,regional_parallel_start=True,
            regional_persistent_modal=True,regional_guard_pressure_reuse=True,regional_compact_rows=True,
            regional_bounded_entries=True,regional_live_support=True,regional_live_surface_prepare=True,
            regional_bounded_refill=True,regional_native_runtime=backend=='native_cuda',regional_exact_capacity_growth=True)
        sc.setdefault('regional_initial_query_slots',512)
        sc.setdefault('regional_initial_working_capacity',64)
        sc.setdefault('regional_strict_native',backend=='native_cuda')
        sc.setdefault('regional_reserve_arenas',True)
        sc.setdefault('regional_pair_capacity_factor',2.)
        sim=create_backend(sc,bodies,'streamed_gpu',policy,batch_capacity,sources)
        if backend=='native_cuda':
            from native_region_adapter import install_refill
            install_refill(sim)
        return sim
    from selected_contact_backend import CONTRACT as selected_contract,CARTESIAN_CONTRACT,ANCHORED_P1_CONTRACT,P1_CONTRACT
    requested=scene.get('common_contact_contract',{})
    if requested in (selected_contract,CARTESIAN_CONTRACT,ANCHORED_P1_CONTRACT,P1_CONTRACT):
        if scene.get('contact_force_convention')!='physical_endpoint' or float(scene['dt'])<=0:
            raise ValueError('Selected contact requires physical endpoint forces and positive time step')
        from selected_contact_backend import create_selected_cpu,create_selected_gpu
        if backend in ('cpu','cpu_columns'):
            return create_selected_cpu(scene,bodies,policy,sources)
        if backend in ('window_gpu','sealed_gpu','separated_gpu','streamed_gpu','mesh_factor_gpu','direct_function_gpu'):
            return create_selected_gpu(scene,bodies,policy,batch_capacity,sources,
                sealed_region=backend in ('sealed_gpu','separated_gpu'),separated_response=backend=='separated_gpu',
                streamed_region=backend=='streamed_gpu',mesh_factor=backend=='mesh_factor_gpu',direct_function=backend=='direct_function_gpu')
        raise NotImplementedError('Unsupported selected-contact execution backend')
    if requested.get('structural_model'):
        raise NotImplementedError('Requested structural/contact contract is not implemented; no legacy fallback')
    contract=ContactContract(**scene.get('common_contact_contract',{}));contract.validate(scene)
    if backend in ('cpu','cpu_columns'):
        from residual_cms import ResidualSurfaceCMS
        from normal_mortar_contact import SmoothedNormalMortarContact
        if backend=='cpu_columns':
            import copy
            from surface_response_source import DenseResidualSource,ResidualColumnView
            sources=sources or [DenseResidualSource(b) for b in bodies]
            bodies=[copy.copy(b) for b in bodies]
            for b,source in zip(bodies,sources):b.R=ResidualColumnView(source)
        sim=ResidualSurfaceCMS(scene,bodies,policy);old=sim.geometry
        sim.geometry=SmoothedNormalMortarContact(bodies,sim.x_offsets,scene['contact_pairs'],old.radius,
            candidate_cache=old.candidate_cache,gap_rule=contract.gap,threads=1)
        sim.response_sources=sources
    elif backend=='dense_gpu':
        from batched_time_simulator import BatchedTimeSurfaceCMS
        sim=BatchedTimeSurfaceCMS(scene,bodies,policy,batch_capacity=batch_capacity)
    elif backend=='window_gpu':
        from windowed_surface_backend import WindowedTimeSurfaceCMS
        sim=WindowedTimeSurfaceCMS(scene,bodies,policy,batch_capacity=batch_capacity,sources=sources)
    elif backend=='sealed_gpu':
        from sealed_region_backend import SealedRegionSurfaceCMS
        sim=SealedRegionSurfaceCMS(scene,bodies,policy,batch_capacity=batch_capacity,sources=sources)
    elif backend=='separated_gpu':
        from separated_region_backend import SeparatedRegionSurfaceCMS
        sim=SeparatedRegionSurfaceCMS(scene,bodies,policy,batch_capacity=batch_capacity,sources=sources)
    else:raise ValueError(backend)
    sim.contact_contract=asdict(contract)
    return sim
