"""Keep residual flexibility and modal dynamics separate in the contact solve.

Resident candidate storage is R[I,I], plus the small regional factor
D[I,:] = Phi[I,:] A_r^-T. No C[S,I], Phi[S,:] A_r^-1, or combined C[I,I]
is stored. The same fixed-step modal, reference and pressure equations apply.
"""
import numpy as np
import cupy as cp
from sealed_region_backend import CompactDeviceBlock,CompactWorkingSet,SealedRegionSurfaceCMS

CODE=r'''
extern "C" __global__ void contact_modal_direction(const double*D,const int*lookup,
 const int*ptr,const int*col,const double*val,const long long*rows,const int*meta,
 double*Y,int modes,int cap){
 int k=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y;
 if(meta[4]||meta[5]||j<meta[3]||j>=meta[2]||k>=modes)return;
 int row=rows[j];double s=0.;
 for(int e=ptr[row];e<ptr[row+1];e++)if(val[e]!=0.){
  int c=lookup[col[e]];if(c>=0)s+=D[(long long)c*modes+k]*val[e];
 }Y[(long long)j*modes+k]=s;
}
extern "C" __global__ void add_modal_region_response(const double*P,const double*Y,
 const int*nodes,const int*count,const int*meta,double*T,int modes,int cap){
 int ii=blockIdx.x,t=threadIdx.x;
 if(meta[4]||meta[5]||ii>=count[0])return;int i=nodes[ii];
 __shared__ double sums[128];
 for(int j=meta[3];j<meta[2];j++){
  double s=0.;for(int k=t;k<modes;k+=blockDim.x)s+=P[(long long)i*modes+k]*Y[(long long)j*modes+k];
  sums[t]=s;__syncthreads();
  for(int d=64;d;d/=2){if(t<d)sums[t]+=sums[t+d];__syncthreads();}
  if(!t)T[(long long)ii*cap+j]+=sums[0];__syncthreads();
 }
}
'''


class SeparatedDeviceBlock(CompactDeviceBlock):
    def initialize_surface_response(self,b):
        super().initialize_surface_response(b)
        self.regional_modal_response=cp.zeros((1,self.P.shape[1]))


class SeparatedWorkingSet(CompactWorkingSet):
    def __init__(self,device,*args,**kw):
        super().__init__(device,*args,**kw)
        module=cp.RawModule(code=CODE)
        self.modal_direction=module.get_function('contact_modal_direction')
        self.modal_response=module.get_function('add_modal_region_response')
        static_response=self.region_k['region_response']
        def response(grid,block,args):
            static_response(grid,block,args)
            i=next(i for i,B in enumerate(device.dynamic) if B.C.data.ptr==args[0].data.ptr)
            B=device.dynamic[i];modes=B.P.shape[1]
            if not modes:return
            _,ptr,col,val,rows,meta,T,_,cap,_,nodes,count=args
            self.modal_direction(((modes+127)//128,self.cap),(128,),
                (B.regional_modal_response,B.window_lookup,ptr,col,val,rows,meta,self.modal_columns[i],np.int32(modes),cap))
            self.modal_response((B.C.shape[0],),(128,),
                (B.P,self.modal_columns[i],nodes,count,meta,T,np.int32(modes),cap))
        self.region_k['region_response']=response
        if device.scene.get('regional_parallel_assembly',False):
            from parallel_region_assembly import install
            install(self,device)
        if device.scene.get('regional_warp_rows',False):
            from parallel_contact_rows import install_working
            install_working(self)
        if device.scene.get('regional_bounded_entries',False):
            from bounded_contact_entries import install_working
            install_working(self,device)
        if device.scene.get('regional_parallel_start',False):
            from parallel_contact_start import install
            install(self)

    def allocate(self,m,nnz,tolerance):
        if self.device.scene.get('regional_parallel_start',False):
            from parallel_contact_start import workspace
            workspace(self,m)
        self.modal_columns=[cp.zeros((self.cap,B.P.shape[1])) for B in self.device.dynamic]
        return super().allocate(m,nnz,tolerance)


class SeparatedRegionSurfaceCMS(SealedRegionSurfaceCMS):
    block_type=SeparatedDeviceBlock

    def __init__(self,*args,**kw):
        super().__init__(*args,**kw)
        self.statistics.update(response_storage='separate_surface_modes_and_residual_regions',
            combined_surface_region_matrix_bytes=0,combined_candidate_matrix_bytes=0,
            full_surface_transformed_modal_factor_bytes=0)

    def region_bytes(self,b,capacity):
        return 8*(capacity*capacity+capacity*b.P.shape[1])

    def allocate_region(self,B,capacity):
        super().allocate_region(B,capacity)
        B.regional_modal_response=cp.zeros((capacity,B.P.shape[1]))

    def write_region_columns(self,B,source,all_ids,missing,occupied,destinations):
        # All I-I cross-patch residual terms come from the global operator.
        values=cp.asarray(source.block(all_ids,missing));ii,jj=cp.asarray(occupied),cp.asarray(destinations)
        B.C[cp.ix_(ii,jj)]=values;B.C[cp.ix_(jj,ii)]=values.T
        # Cache a regional factor only; A_r is constant for this fixed-step model.
        B.regional_modal_response[jj]=B.P[cp.asarray(missing)]@B.inverse.T

    def rebuild(self,refresh=True):
        if not isinstance(self.graph_working,SeparatedWorkingSet):self.graph_working=SeparatedWorkingSet(self)
        return super().rebuild(refresh)

    def storage_ledger(self):
        """Explicit array ownership, not a process or device-pool peak counter."""
        rows=[]
        for b,B,op in zip(self.bodies,self.dynamic,self.sealed_device):
            # B.C is the legacy API name for this variant's residual-only block.
            assert B.C.shape[0]==B.C.shape[1] and B.modal_response.size==0
            shared=getattr(op,'modes',None)
            if op.modal_count:assert shared.data.ptr==B.P.data.ptr
            rows.append(dict(surface_dofs=len(B.P),modes=B.P.shape[1],candidate_dofs=len(B.compact_ids),
                capacity=B.C.shape[0],surface_modes_bytes=B.P.nbytes,residual_region_bytes=B.C.nbytes,
                regional_modal_factor_bytes=B.regional_modal_response.nbytes,
                global_residual_representation_bytes=op.stored_bytes,
                effective_modal_inverse_bytes=B.inverse.nbytes,
                combined_surface_region_matrix_bytes=0,combined_candidate_matrix_bytes=0))
        return dict(bodies=rows,scope='listed CUDA arrays; excludes other modal/frame matrices, geometry, workspaces and memory-pool reservations',
            online_full_order_factor_solves=0,whole_surface_geometry_recovery=True)
