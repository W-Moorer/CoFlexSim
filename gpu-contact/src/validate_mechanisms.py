"""Portable paths and compact evidence helpers (no local research/job controllers)."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def guard():
    if not (ROOT/'coflex.py').is_file():raise RuntimeError('Run from the CoFlexSim source distribution')
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()
def plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,default=plain)+'\n',encoding='utf-8')
def scene_for(case):
    from public_runtime import load_scene
    return load_scene(case)
