"""Platform loader for the unchanged CUDA conditional graph kernels."""
import ctypes as ct,hashlib,os,subprocess
from pathlib import Path
_lib=None

def library():
    global _lib
    if _lib is not None:return _lib
    from public_build import compiler_environment
    compiler_environment()
    src=Path(__file__).resolve().parent
    code=(src/'device_graph.cu').read_text()+'\n'+(src/'trajectory_graph.cu').read_text().replace('#include "device_graph.cu"','')
    if os.name!='nt':code=code.replace('__declspec(dllexport)','')
    folder=src.parent/'cache/graph';folder.mkdir(parents=True,exist_ok=True)
    identity=hashlib.sha256((code+os.name).encode()).hexdigest()[:16]
    binary=folder/(identity+('.dll' if os.name=='nt' else '.so'))
    if not binary.exists():
        cu=folder/(identity+'.cu');cu.write_text(code)
        cmd=['nvcc','-shared','-O3','-std=c++17','-arch=sm_86',str(cu),'-o',str(binary)]
        cmd+=['-allow-unsupported-compiler','-Xcompiler','/MD'] if os.name=='nt' else ['-Xcompiler','-fPIC']
        subprocess.run(cmd,check=True,cwd=folder)
    _lib=ct.CDLL(str(binary));p=ct.c_void_p;i=ct.c_int
    specs={'dc_error':([i],ct.c_char_p),'dc_begin':([p],i),'dc_end':([p,ct.POINTER(p)],i),
        'dc_launch':([p,p],i),'dc_destroy':([p,p],i),'dc_instantiate':([p,ct.POINTER(p)],i),
        'dc_while':([p,p,i,i,i,i,ct.POINTER(p),ct.POINTER(p)],i),
        'dc_nested':([ct.POINTER(p),p,p,i,i,ct.POINTER(p),ct.POINTER(p)],i),
        'dc_trajectory':([ct.POINTER(p),p,p,p,ct.POINTER(p),ct.POINTER(p)],i),'dc_trajectory_upload':([p,p],i)}
    for name,(args,ret) in specs.items():
        f=getattr(_lib,name);f.argtypes=args;f.restype=ret
    return _lib
