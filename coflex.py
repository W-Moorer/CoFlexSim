"""CoFlexSim: one entry point for all examples and execution backends."""
import os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'gpu-contact/cache'
for key,sub in {'TEMP':'temp','TMP':'temp','CUDA_CACHE_PATH':'cuda','CUPY_CACHE_DIR':'cupy','MPLCONFIGDIR':'matplotlib'}.items():
    p=CACHE/sub;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CONTACT_CAPTURE_WARMUP_PASSES']='2'
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'gpu-contact/src'),str(CACHE/'extensions')]

def main():
    import argparse
    p=argparse.ArgumentParser(description='CoFlexSim unified contact examples')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('fetch-data',help='Download and verify release example data')
    b=sub.add_parser('build');b.add_argument('--eigen-root',type=Path)
    sub.add_parser('check',help='Check imports, native libraries and CUDA device')
    for name in ('prepare','run'):
        a=sub.add_parser(name);a.add_argument('case',choices=['spur','helical','cam'])
        if name=='prepare':
            a.add_argument('--bank-budget-gib',type=float,default=12.)
        else:
            a.add_argument('--backend',choices=['native_cuda','host_cuda','cpu_columns'],default='native_cuda')
            a.add_argument('--steps',type=int,help='Omit for configured short trajectory; --full for long')
            a.add_argument('--full',action='store_true');a.add_argument('--output',type=Path)
            a.add_argument('--prepared',type=Path,help='Trusted, locally generated preparation manifest')
            a.add_argument('--restart',type=Path,help='Accepted HHT state archive for a matched short window')
            a.add_argument('--memory',action='store_true',help='Separate memory diagnostic, not clean timing')
    a=p.parse_args()
    if a.command=='build':
        from public_build import build
        build(a.eigen_root);return
    from public_runtime import fetch_data,prepare,run,check
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        if a.command=='fetch-data':fetch_data()
        elif a.command=='prepare':prepare(a.case,a.bank_budget_gib)
        elif a.command=='run':run(a)
        else:check()

if __name__=='__main__':main()
