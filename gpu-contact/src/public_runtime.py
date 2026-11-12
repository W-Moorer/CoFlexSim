"""Reproducible scene preparation and one backend factory for all three cases."""
import argparse,copy,gc,hashlib,json,os,time,urllib.request,zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'gpu-contact';DATA=BASE/'data'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def cast(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=cast)+'\n',encoding='utf-8')

def save_object(path,obj):
    """Array archive without pickle or executable object deserialization."""
    arrays={}
    def enc(v):
        if isinstance(v,np.ndarray):
            key='a'+str(len(arrays));arrays[key]=v;return {'$array':key}
        if isinstance(v,SimpleNamespace):return {'$namespace':enc(vars(v))}
        if isinstance(v,dict):return {k:enc(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [enc(x) for x in v]
        if isinstance(v,np.generic):return v.item()
        return v
    arrays['metadata']=np.asarray(json.dumps(enc(obj)))
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**arrays)

def load_object(path):
    with np.load(path,allow_pickle=False) as archive:
        def dec(v):
            if isinstance(v,dict):
                if '$array' in v:return archive[v['$array']].copy()
                if '$namespace' in v:return SimpleNamespace(**dec(v['$namespace']))
                return {k:dec(x) for k,x in v.items()}
            if isinstance(v,list):return [dec(x) for x in v]
            return v
        return dec(json.loads(str(archive['metadata'])))

def config(case):return json.loads((ROOT/f'examples/{case}/case.json').read_text())

def fetch_data():
    manifest=json.loads((ROOT/'examples/data_manifest.json').read_text())
    items=manifest.get('assets',[manifest])
    DATA.mkdir(exist_ok=True,parents=True)
    for item in items:
        path=DATA/item['asset']
        if not path.exists() or sha(path)!=item['sha256']:
            tag=item.get('tag',manifest['tag'])
            url=f"https://github.com/W-Moorer/CoFlexSim/releases/download/{tag}/{item['asset']}"
            print('Downloading',item['asset'],flush=True)
            partial=path.with_suffix('.download')
            with urllib.request.urlopen(url,timeout=180) as src,partial.open('wb') as dst:
                import shutil
                shutil.copyfileobj(src,dst,8*2**20)
            if sha(partial)!=item['sha256']:raise RuntimeError('Release data checksum mismatch')
            partial.replace(path)
        if path.suffix=='.zip':
            with zipfile.ZipFile(path) as z:
                for name in z.namelist():
                    if not (DATA/name).resolve().is_relative_to(DATA.resolve()):raise ValueError('Unsafe archive path')
                z.extractall(DATA)
        print('Verified',item['asset'],flush=True)

def load_scene(case):
    p=DATA/case/'scene.json'
    if not p.exists():raise FileNotFoundError('Run python coflex.py fetch-data first')
    cfg=config(case)
    if sha(p)!=cfg['scene_sha256']:raise ValueError('Example scene checksum changed')
    scene=json.loads(p.read_text())
    for body in scene['bodies']:
        for key,value in body.items():
            if isinstance(value,list):body[key]=np.asarray(value)
    from selected_contact_backend import CARTESIAN_CONTRACT
    scene['common_contact_contract']=CARTESIAN_CONTRACT.copy()
    return scene

def prepare(case,budget_gib=12.):
    from coflex_core import Body,DEFAULT_POLICY
    from quasistatic_pd import coflex_pd_native
    from fast_modal_precompute import select_modes,prepare_runtime
    from surface_response_source import SparseResidualSource
    from host_surface_library import HostSurfaceLibrary
    cfg=config(case);scene=load_scene(case);folder=preparation_folder(case,cfg);folder.mkdir(parents=True,exist_ok=True)
    target=folder/'manifest.json'
    if target.exists():
        print('Existing completed preparation:',target);return
    records=[];remaining=int(budget_gib*2**30)
    for j,spec in enumerate(scene['bodies']):
        tick=time.perf_counter();bodyfile=folder/f'runtime_body{j}.npz'
        FE=Body(spec,scene['dt'],DEFAULT_POLICY,False,native_module=coflex_pd_native)
        seed=DATA/case/f'runtime_body{j}.npz'
        if bodyfile.exists():
            b=load_object(bodyfile);selection={'reused_local_runtime':True,'sha256':sha(bodyfile)}
        elif seed.exists():
            b=load_object(seed);selection={'reused_release_runtime':True,'sha256':sha(seed)}
        elif 'selected_indices' in cfg:
            raise FileNotFoundError('The fine-gear example requires its verified runtime release asset; run fetch-data.')
        else:
            hints=cfg.get('modal_candidate_hints',[512]*len(scene['bodies']))
            threads=cfg.get('preparation_threads',4)
            P,selection=select_modes(FE,scene['dt'],candidate_hint=hints[j],threads=threads)
            b,P=prepare_runtime(FE,P,threads=threads)
            del P
        save_object(bodyfile,b);operator_s=time.perf_counter()-tick
        source=SparseResidualSource(FE,b);identity={'scene_sha256':cfg['scene_sha256'],'body':j,'runtime_sha256':sha(bodyfile)}
        tick=time.perf_counter()
        bank=HostSurfaceLibrary.build(source,file_path=folder/f'residual{j}.f64',source_identity=identity,
            budget_bytes=remaining,progress=lambda done,total:print(f'{case} body {j}: {done}/{total}',flush=True) if done==total or done%1920==0 else None)
        if not hasattr(bank,'far_column_bound'):bank.prepare_far_bounds(b.spec['contact_faces']);bank.save_catalog()
        ids=np.unique(np.linspace(0,bank.n-1,min(31,bank.n),dtype=int));ref=source.columns(ids)[ids]
        error=float(np.linalg.norm(bank.block(ids,ids)-ref)/max(np.linalg.norm(ref),1e-30))
        if error>1e-10:raise RuntimeError(('Residual bank validation failed',error))
        rec={'runtime':bodyfile.name,'bank':f'residual{j}.f64','identity':identity,'operator_s':operator_s,
            'bank_and_bounds_s':time.perf_counter()-tick,'bank_bytes':bank.backing.nbytes,'selection':selection,'bank_check_relative':error}
        if 'selected_indices' in cfg:
            from selected_residual_region import SelectedResidualRegion
            from tight_residual_certificate import prepare as certificate
            op=SelectedResidualRegion(bank,b,cfg['selected_indices'][j]);cert=folder/f'far{j}.npz'
            rec['selected_indices']=cfg['selected_indices'][j];rec['certificate']=cert.name
            rec['selected_certificate']=certificate(op,b.spec['contact_faces'],path=cert)
            del op
        records.append(rec);remaining-=bank.backing.nbytes
        del source,bank,FE,b;gc.collect()
    save(target,{'case':case,'scene_sha256':cfg['scene_sha256'],'bodies':records,'offline_only':True})
    print('Prepared',target)

def preparation_folder(case,cfg=None):
    """Reuse a matching preparation; isolate inputs with a new mesh identity."""
    cfg=config(case) if cfg is None else cfg
    legacy=DATA/case/'prepared'
    manifest=legacy/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()).get('scene_sha256')==cfg['scene_sha256']:
        return legacy
    return legacy/cfg['scene_sha256'][:12]


def load_prepared(case,path=None):
    from host_surface_library import HostSurfaceLibrary
    scene=load_scene(case);path=Path(path) if path else preparation_folder(case)/'manifest.json'
    if not path.exists():raise FileNotFoundError('Run python coflex.py prepare '+case)
    info=json.loads(path.read_text());folder=path.parent
    if info['scene_sha256']!=config(case)['scene_sha256']:raise ValueError('Preparation belongs to a different scene')
    bodies=[];banks=[]
    for i,r in enumerate(info['bodies']):
        b=load_object(folder/r['runtime']);s=HostSurfaceLibrary.open(folder/r['bank'],r['identity'])
        # Contact faces come from the declared runtime surface, including the
        # fine-gear flank-only mask. No scene-name branch in the solver.
        scene['bodies'][i]['contact_faces']=b.original_nodes[b.spec['contact_faces']]
        if 'selected_indices' in r:
            from selected_residual_region import SelectedResidualRegion
            from fast_modal_precompute import restrict_runtime
            from tight_residual_certificate import prepare as certificate
            op=SelectedResidualRegion(s,b,r['selected_indices'])
            certificate(op,b.spec['contact_faces'],path=folder/r['certificate'])
            b=restrict_runtime(b,r['selected_indices']);s=op
        bodies.append(b);banks.append(s)
    return scene,bodies,banks,info

def run(args):
    import platform,psutil
    cfg=config(args.case);steps=args.steps if args.steps is not None else (round(cfg['duration_s']/cfg['dt_s']) if args.full else cfg['short_steps'])
    if steps<=0:raise ValueError('steps must be positive')
    target=args.output or ROOT/f'outputs/{args.case}_{args.backend}_{steps}.json'
    if target.exists():raise FileExistsError('Choose a new --output to preserve existing evidence: '+str(target))
    tick=time.perf_counter();scene,bodies,banks,info=load_prepared(args.case,args.prepared);load_s=time.perf_counter()-tick
    scene.update(steps=steps,response_window_bytes=cfg['region_budget_MiB']*2**20,regional_initial_working_capacity=64,
        regional_initial_query_slots=512,regional_pair_capacity_factor=cfg['pair_capacity_factor'],regional_refill_slot_bytes=4*2**20)
    policy=cfg.get('solver_policy',{});cp=None;hook=None
    if args.backend!='cpu_columns':
        import cupy as cp
        if args.memory:
            class Peak(cp.cuda.MemoryHook):
                name='coflex_public_peak'
                def __init__(self):self.live={};self.current=0;self.peak=0
                def malloc_postprocess(self,**kw):
                    if kw['pmem_id']:
                        self.live[kw['pmem_id']]=kw['mem_size'];self.current+=kw['mem_size'];self.peak=max(self.peak,self.current)
                def free_postprocess(self,**kw):self.current-=self.live.pop(kw['pmem_id'],0)
            hook=Peak();hook.__enter__()
    from common_contact_backend import create_backend
    tick=time.perf_counter();sim=create_backend(scene,bodies,args.backend,policy,batch_capacity=cfg['batch_capacity'],sources=banks)
    if getattr(args,'restart',None):
        with np.load(args.restart,allow_pickle=False) as state:
            for key in ('d','v','acc','s','p','old_nonmass','x'):
                if getattr(sim,key).shape!=state[key].shape:raise ValueError('Restart shape differs: '+key)
                getattr(sim,key)[:]=cp.asarray(state[key]) if cp else state[key]
            sim.t=float(state['time'])
        sim.old_nonmass[:]=sim.nonmass(sim.d,sim.v,sim.t)
        if cp:sim.window_ready=False;sim.rebuild(refresh=True)
    if args.backend=='native_cuda':
        from native_runtime_adapter import NativeRuntime
        sim.native_runtime=NativeRuntime(sim)
    if cp:cp.cuda.get_current_stream().synchronize()
    init_s=time.perf_counter()-tick;rows=[];online=0.
    while len(rows)<steps:
        tick=time.perf_counter()
        if cp:
            rr,_=sim.advance_batch(min(cfg['batch_capacity'],steps-len(rows)));cp.cuda.get_current_stream().synchronize()
        else:
            z=sim.step();z['contact_moment_about_reference_Nm']=[np.cross(sim.x[lo:hi]-b.center,sim.last_nodal_force[lo:hi]).sum(0).tolist() for b,lo,hi in zip(bodies,sim.x_offsets[:-1],sim.x_offsets[1:])];rr=[z]
        online+=time.perf_counter()-tick
        rows.extend({k:z[k] for k in ('t','force','refs','contact_moment_about_reference_Nm','pen_um')} for z in rr)
        if len(rows)%1024==0 or len(rows)==steps:print(args.case,args.backend,len(rows),steps,f'{online:.3f} s',flush=True)
    proc=psutil.Process();mem=proc.memory_info()
    report={'status':'complete','case':args.case,'backend':args.backend,'steps':steps,'dt_s':scene['dt'],'end_s':sim.t,
        'input_load_s':load_s,'initialization_s':init_s,'online_s':online,'offline_preparation':info,
        'platform':platform.platform(),'rows':rows,'peak_RSS_bytes':getattr(mem,'peak_wset',mem.rss),
        'timing_scope':'synchronized online solve including region refills; offline/load/init/output separate',
        'memory_instrumented':bool(args.memory)}
    if cp:
        report['GPU']=cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        report['GPU_pool_reserved_bytes']=cp.get_default_memory_pool().total_bytes()
        if hasattr(sim,'storage_ledger'):report['storage']=sim.storage_ledger()
    if args.backend=='native_cuda':report['native_runtime']=sim.native_runtime.ledger();report['native_staging']=sim.refill_staging.ledger()
    if hook:
        hook.__exit__(None,None,None);extra=0
        if args.backend=='native_cuda':extra=report['native_runtime']['region']['device_bytes']+report['native_staging']['device_staging_and_workspace_bytes']
        report['owned_GPU_peak_upper_bound_bytes']=hook.peak+extra
    # Local preparation manifests may refer to local files; never publish raw
    # run logs without the compact evidence export described in docs.
    save(target,report);print('Saved',target)

def check():
    import coflex_native,coflex_pd_native,coflex_mortar_native,cupy as cp
    from portable_graph import library
    from native_region_adapter import library as native_library
    library();native_library()
    print(json.dumps({'python':__import__('sys').version.split()[0],'GPU':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        'CUDA':cp.cuda.runtime.runtimeGetVersion(),'mechanics':'loaded','graph_bridge':'loaded','native_runtime':'loaded'},indent=2))
