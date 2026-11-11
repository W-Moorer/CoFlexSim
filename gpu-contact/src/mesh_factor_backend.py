"""Exact resident-factor execution of the shared surface CMS/contact equations."""
import numpy as np
import cupy as cp
from device_simulator import DeviceBlock
from windowed_surface_backend import WindowWorkingSet,WindowContact,WindowedTimeSurfaceCMS
from aliased_working import AliasedWorkingSet
from sealed_region_backend import SealedContact
from mesh_residual_gpu import matmul
from mesh_supernode_gpu import DeviceMeshSupernode

CODE=r'''
extern "C" __global__ void pack_rhs(const int*ptr,const int*col,const double*val,
 const long long*rows,const int*meta,double*F,int n,int cap){
 int j=blockIdx.x,t=threadIdx.x;
 if(meta[4]||meta[5]||j<meta[3]||j>=meta[2])return;
 int r=rows[j];for(int e=ptr[r]+t;e<ptr[r+1];e+=blockDim.x)
  if(val[e]!=0.)atomicAdd(F+(long long)col[e]*cap+j,val[e]);
}
extern "C" __global__ void collect_response(const double*X,const int*nodes,const int*count,
 const int*meta,double*T,int cap){
 int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y;
 if(meta[4]||meta[5]||i>=count[0]||j<meta[3]||j>=meta[2])return;
 T[(long long)i*cap+j]=X[(long long)nodes[i]*cap+j];
}
'''

class MeshFactorBlock(DeviceBlock):
    def initialize_surface_response(self,b):
        self.R=None
        # API handle only: it is never interpreted as a response matrix.
        self.C=cp.zeros((len(self.P),1))

class MeshFactorWorking(WindowWorkingSet):
    def __init__(self,device,*args,**kw):
        super().__init__(device,*args,**kw)
        module=cp.RawModule(code=CODE)
        self.pack_rhs=module.get_function('pack_rhs');self.collect=module.get_function('collect_response')
        def response(grid,block,args):
            C,ptr,col,val,ids,meta,T,_,cap,_,nodes,count=args
            i=next(i for i,B in enumerate(device.dynamic) if B.C.data.ptr==C.data.ptr)
            op=device.sealed_device[i];B=device.dynamic[i];w=op.work[int(cap)]
            F=w['f'];F.fill(0)
            self.pack_rhs((int(cap),),(128,),(ptr,col,val,ids,meta,F,np.int32(op.n),cap))
            # Combine static modal subtraction and dynamic modal addition before
            # the surface mapping. No second full-surface modal temporary.
            result=op.apply(F,meta=meta,modal_inverse=op.contact_inverse)
            self.collect(((op.n+127)//128,int(cap)),(128,),(result,nodes,count,meta,T,cap))
        self.region_k['region_response']=response
        if device.scene.get('regional_parallel_assembly',False):
            from parallel_region_assembly import CODE as PARALLEL_CODE
            contact=cp.RawModule(code=PARALLEL_CODE).get_function('contact_columns')
            def assemble(grid,block,args):contact((grid[0],int(args[8])),block,args)
            self.region_k['region_contact']=assemble
        if device.scene.get('regional_parallel_start',False):
            from parallel_contact_start import install
            install(self)

    def allocate(self,m,nnz,tolerance):
        for op in self.device.sealed_device:op.reserve(self.cap)
        if self.device.scene.get('regional_parallel_start',False):
            from parallel_contact_start import workspace
            workspace(self,m)
        return super().allocate(m,nnz,tolerance)

    def enqueue_prepare(self,b,keys,valid=None):
        # Full designated surface is covered by the factor, so no region miss.
        return AliasedWorkingSet.enqueue_prepare(self,b,keys,valid)

class MeshFactorContact(SealedContact):
    pass

class MeshFactorSurfaceCMS(WindowedTimeSurfaceCMS):
    block_type=MeshFactorBlock
    def __init__(self,*args,**kwargs):
        self.sealed_device=[]
        super().__init__(*args,**kwargs)
        self.statistics.update(response_storage='resident_exact_mesh_factor',host_flexibility_refills=0,
            online_full_order_factorizations=0,online_sparse_factor_actions=True)

    def refresh_response_window(self):
        self.window_ready=True
        return False

    def rebuild(self,refresh=True):
        if not self.sealed_device:
            self.sealed_device=[DeviceMeshSupernode(s,B.P) for s,B in zip(self.sources,self.dynamic)]
            for op,B in zip(self.sealed_device,self.dynamic):
                op.contact_inverse=op.Kinv-B.inverse
                op.parameter_bytes+=op.contact_inverse.nbytes
        if not isinstance(self.resident,MeshFactorContact):self.resident=MeshFactorContact(self)
        if self.scene.get('regional_compact_rows',False):
            from compact_contact_rows import install
            install(self.geometry)
        return super().rebuild(refresh)

    def storage_ledger(self):
        return dict(bodies=[op.storage() for op in self.sealed_device],
                    scope='resident factor, shared surface modes and query/analysis buffers; full solve peak measured separately',
                    host_library_refills=0,dense_surface_or_region_matrix_bytes=0,
                    all_designated_surface_covered=True,whole_surface_geometry_recovery=True)
