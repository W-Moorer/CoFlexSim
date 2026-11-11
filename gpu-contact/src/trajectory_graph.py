"""A separate native three-level graph bridge; v5 remains unchanged."""
import ctypes as ct
import cupy as cp
import hashlib,os,subprocess
from pathlib import Path
from conditional_graph import NativeGraph,checked
from validate_mechanisms import ROOT,guard
_lib=None

def library():
    from portable_graph import library as load
    return load()

def trajectory_graph(parts,outer,inner,run):
    graph=NativeGraph();graph.refs=[*parts,outer,inner,run,library()]
    values=(ct.c_void_p*6)(*(p.graph.value for p in parts))
    checked(library().dc_trajectory(values,outer.data.ptr,inner.data.ptr,run.data.ptr,ct.byref(graph.graph),ct.byref(graph.exec)))
    # Upload is initialization, with no numerical steps executed or discarded.
    checked(library().dc_trajectory_upload(graph.exec,cp.cuda.get_current_stream().ptr))
    return graph
