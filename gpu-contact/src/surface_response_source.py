"""Exact whole-body residual columns, independent of contact/integrator backend.

The sparse provider stores a stiffness factor and bounded column cache. It never
constructs a full surface inverse. Online cache misses are timed online.
"""
from collections import OrderedDict
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla


class DenseResidualSource:
    def __init__(self,body):
        self.body=body;self.n=body.P.shape[0]
        self.statistics=dict(source='existing_dense_host_reference',column_requests=0,miss_columns=0,query_s=0.)
    def columns(self,ids):
        ids=np.asarray(ids,dtype=int);start=time.perf_counter()
        out=np.zeros((self.n,len(ids))) if self.body.R is None else np.asarray(self.body.R[:,ids])
        self.statistics['column_requests']+=len(ids);self.statistics['query_s']+=time.perf_counter()-start
        return out
    def block(self,rows,cols):
        start=time.perf_counter()
        out=np.zeros((len(rows),len(cols))) if self.body.R is None else self.body.R[np.ix_(rows,cols)]
        self.statistics['block_requests']=self.statistics.get('block_requests',0)+1
        self.statistics['query_s']+=time.perf_counter()-start
        return out


class SparseResidualSource:
    def __init__(self,FE,runtime,cache_bytes=64*2**20,batch=32):
        start=time.perf_counter();self.n=runtime.P.shape[0]
        self.factor=sla.splu(FE.K.tocsc());self.free_count=len(FE.free)
        inverse=np.full(FE.X.size,-1,dtype=np.int64);inverse[FE.free]=np.arange(self.free_count)
        self.lookup=inverse[runtime.original_surface_rows]
        self.P=np.asarray(runtime.P);self.modal_inverse=la.inv(runtime.K) if runtime.K.size else np.zeros((0,0))
        self.cache=OrderedDict();self.capacity=max(1,int(cache_bytes)//max(8*self.n,1));self.batch=batch
        self.statistics=dict(source='whole_body_sparse_K_factor',column_requests=0,miss_columns=0,
            query_s=0.,factorization_s=time.perf_counter()-start,cache_capacity_columns=self.capacity,
            full_surface_matrix_constructed=False,reference_K_dofs=self.free_count)
    def columns(self,ids):
        start=time.perf_counter();ids=np.asarray(ids,dtype=np.int64)
        if np.any(ids<0) or np.any(ids>=self.n):raise IndexError('Surface response column outside model')
        out=np.empty((self.n,len(ids)));missing=[];places=[]
        for j,i in enumerate(ids):
            if int(i) in self.cache:
                out[:,j]=self.cache[int(i)];self.cache.move_to_end(int(i))
            else:missing.append(int(i));places.append(j)
        for lo in range(0,len(missing),self.batch):
            request=np.asarray(missing[lo:lo+self.batch]);where=places[lo:lo+self.batch]
            rhs=np.zeros((self.free_count,len(request)),order='F');mapped=self.lookup[request]
            valid=mapped>=0;rhs[mapped[valid],np.flatnonzero(valid)]=1.
            x=self.factor.solve(rhs);response=np.zeros((self.n,len(request)))
            valid_surface=self.lookup>=0;response[valid_surface]=x[self.lookup[valid_surface]]
            response-=self.P@(self.modal_inverse@self.P[request].T)
            out[:,where]=response
            for j,i in enumerate(request):
                self.cache[int(i)]=response[:,j].copy()
                while len(self.cache)>self.capacity:self.cache.popitem(last=False)
        self.statistics['column_requests']+=len(ids);self.statistics['miss_columns']+=len(missing)
        self.statistics['query_s']+=time.perf_counter()-start
        self.statistics['cache_bytes']=len(self.cache)*self.n*8
        return out


class ResidualColumnView:
    """CPU adapter for the existing contact-region solver; no global dense array."""
    def __init__(self,source):self.source=source;self.shape=(source.n,source.n)
    def __getitem__(self,index):
        rows,cols=index
        if isinstance(cols,slice):raise ValueError('Request explicit contact columns, not the full inverse')
        ids=np.asarray(cols,dtype=int).ravel()
        if not isinstance(rows,slice) and hasattr(self.source,'block'):
            return self.source.block(np.asarray(rows,dtype=int).ravel(),ids)
        out=self.source.columns(ids)
        if isinstance(rows,slice):return out[rows]
        return out[np.asarray(rows,dtype=int).ravel()]
