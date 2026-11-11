"""Build local mechanics extensions and the C++/CUDA scheduler."""
import os,sys,subprocess,shutil,hashlib,urllib.request,tarfile,io
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'gpu-contact';CACHE=BASE/'cache'

def compiler_environment():
    if os.name!='nt':return
    cl=shutil.which('cl.exe')
    if not cl:raise RuntimeError('Open an x64 Native Tools Command Prompt for Visual Studio, then rerun.')
    vc=Path(cl).parents[3]
    sdk=Path(os.environ.get('ProgramFiles(x86)','C:/Program Files (x86)'))/'Windows Kits/10'
    candidates=sorted((sdk/'bin').glob('10.*'),reverse=True)
    version=next((p.name for p in candidates if (p/'x64/rc.exe').is_file()),None)
    if not version:raise RuntimeError('Windows 10/11 SDK is required')
    os.environ['PATH']=str(sdk/'bin'/version/'x64')+os.pathsep+os.environ['PATH']
    os.environ['INCLUDE']=';'.join(map(str,[vc/'include']+[sdk/'Include'/version/x for x in ('ucrt','shared','um')]))
    os.environ['LIB']=';'.join(map(str,[vc/'lib/x64']+[sdk/'Lib'/version/x/'x64' for x in ('ucrt','um')]))

def build(eigen=None):
    import pybind11
    compiler_environment()
    if eigen is None:
        eigen=CACHE/'deps/eigen-3.4.0'
        if not (eigen/'Eigen/Core').is_file():
            url='https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz'
            data=urllib.request.urlopen(url,timeout=120).read()
            if hashlib.sha256(data).hexdigest()!='8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72':
                raise RuntimeError('Eigen archive checksum differs')
            (CACHE/'deps').mkdir(parents=True,exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
                archive.extractall(CACHE/'deps',filter='data')
    eigen=eigen.resolve()
    if not (eigen/'Eigen/Core').is_file():raise FileNotFoundError(eigen/'Eigen/Core')
    generator='NMake Makefiles' if os.name=='nt' else 'Unix Makefiles'
    tasks=[(BASE/'mechanics',CACHE/'extensions',[
        '-DEIGEN_ROOT='+str(eigen),'-Dpybind11_DIR='+pybind11.get_cmake_dir(),'-DPython_EXECUTABLE='+sys.executable]),
        (BASE/'native',CACHE/'native_cpp',[])]
    for src,out,extra in tasks:
        subprocess.run(['cmake','-S',str(src),'-B',str(out),'-G',generator,'-DCMAKE_BUILD_TYPE=Release',*extra],check=True)
        subprocess.run(['cmake','--build',str(out),'--parallel','2'],check=True)
    from portable_graph import library
    library()
    print('Mechanics, C++/CUDA scheduler and conditional-graph bridge built.')
