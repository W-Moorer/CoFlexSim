"""Exact supernode substitution using only kernels inside conditional graphs.

Dense diagonal-panel inverses are computed offline. Inter-panel contributions
remain complete. This changes execution/storage, not the residual operator.
"""
import time
import numpy as np
import scipy.linalg as la
import cupy as cp
from mesh_residual_gpu import matmul

CODE=r'''
__device__ bool active(int j,const int*meta){return !meta[4]&&!meta[5]&&j>=meta[3]&&j<meta[2];}
__device__ double warp_sum(double x){for(int d=16;d;d/=2)x+=__shfl_down_sync(0xffffffff,x,d);return x;}
extern "C" __global__ void diagonal_panel(const int*list,const int*start,const int*size,
 const long long*ip,const double*iv,const double*rhs,double*x,int width,const int*meta,int transpose){
 int g=list[blockIdx.x],n=size[g],s=start[g];const double*A=iv+ip[g];
 int lane=threadIdx.x%32;
 for(int q=threadIdx.x/32;q<n*width;q+=blockDim.x/32){int r=q/width,j=q%width;if(!active(j,meta))continue;
  double v=0.;for(int k=lane;k<n;k+=32)v+=A[transpose?(long long)k*n+r:(long long)r*n+k]*rhs[(long long)(s+k)*width+j];
  v=warp_sum(v);if(!lane)x[(long long)(s+r)*width+j]=v;}
}
extern "C" __global__ void forward_panel(const int*list,const int*start,const int*size,
 const int*rp,const int*rows,const long long*vp,const double*val,const double*x,double*rhs,int width,const int*meta){
 int g=list[blockIdx.x],n=size[g],s=start[g],nr=3*(rp[g+1]-rp[g]);const double*A=val+vp[g];
 for(int q=threadIdx.x;q<nr*width;q+=blockDim.x){int r=q/width,j=q%width;if(!active(j,meta))continue;
  double v=0.;for(int k=0;k<n;k++)v+=A[(long long)r*n+k]*x[(long long)(s+k)*width+j];
  int target=3*rows[rp[g]+r/3]+r%3;atomicAdd(rhs+(long long)target*width+j,-v);}
}
extern "C" __global__ void backward_panel(const int*list,const int*start,const int*size,
 const int*rp,const int*rows,const long long*vp,const double*val,const double*x,double*rhs,int width,const int*meta){
 int g=list[blockIdx.x],n=size[g],s=start[g],nr=3*(rp[g+1]-rp[g]);const double*A=val+vp[g];
 int lane=threadIdx.x%32;
 for(int q=threadIdx.x/32;q<n*width;q+=blockDim.x/32){int r=q/width,j=q%width;if(!active(j,meta))continue;
  double v=0.;
  for(int k=lane;k<nr;k+=32){int target=3*rows[rp[g]+k/3]+k%3;v+=A[(long long)k*n+r]*x[(long long)target*width+j];}
  v=warp_sum(v);if(!lane)rhs[(long long)(s+r)*width+j]=x[(long long)(s+r)*width+j]-v;}
}
extern "C" __global__ void scatter_input(const double*f,const int*ids,double*x,int n,int width){
 int q=blockIdx.x*blockDim.x+threadIdx.x;if(q<n*width){int i=q/width,j=q%width;x[(long long)ids[i]*width+j]=f[q];}}
extern "C" __global__ void gather_output(const double*x,const int*ids,double*out,int n,int width){
 int q=blockIdx.x*blockDim.x+threadIdx.x;if(q<n*width){int i=q/width,j=q%width;out[q]=x[(long long)ids[i]*width+j]-out[q];}}
extern "C" __global__ void level_diagonal(const int*groups,const int*local,int count,const int*start,const int*size,
 const long long*ip,const double*iv,const double*rhs,double*x,int width,const int*meta,int transpose){
 if(meta[4]||meta[5])return;int nc=meta[2]-meta[3],lane=threadIdx.x%32;
 for(int q=(blockIdx.x*blockDim.x+threadIdx.x)/32;q<count*nc;q+=gridDim.x*(blockDim.x/32)){
  int a=q/nc,j=meta[3]+q%nc,g=groups[a],r=local[a],n=size[g],s=start[g];const double*A=iv+ip[g];
  double v=0.;for(int k=lane;k<n;k+=32)v+=A[transpose?(long long)k*n+r:(long long)r*n+k]*rhs[(long long)(s+k)*width+j];
  v=warp_sum(v);if(!lane)x[(long long)(s+r)*width+j]=v;
 }
}
extern "C" __global__ void level_forward(const int*groups,const int*local,int count,const int*start,const int*size,
 const int*rp,const int*rows,const long long*vp,const double*val,const double*x,double*rhs,int width,const int*meta){
 if(meta[4]||meta[5])return;int nc=meta[2]-meta[3];
 for(int q=blockIdx.x*blockDim.x+threadIdx.x;q<count*nc;q+=gridDim.x*blockDim.x){
  int a=q/nc,j=meta[3]+q%nc,g=groups[a],r=local[a],n=size[g],s=start[g];const double*A=val+vp[g];
  double v=0.;for(int k=0;k<n;k++)v+=A[(long long)r*n+k]*x[(long long)(s+k)*width+j];
  int target=3*rows[rp[g]+r/3]+r%3;atomicAdd(rhs+(long long)target*width+j,-v);
 }
}
extern "C" __global__ void level_backward(const int*groups,const int*local,int count,const int*start,const int*size,
 const int*rp,const int*rows,const long long*vp,const double*val,const double*x,double*rhs,int width,const int*meta){
 if(meta[4]||meta[5])return;int nc=meta[2]-meta[3],lane=threadIdx.x%32;
 for(int q=(blockIdx.x*blockDim.x+threadIdx.x)/32;q<count*nc;q+=gridDim.x*(blockDim.x/32)){
  int a=q/nc,j=meta[3]+q%nc,g=groups[a],r=local[a],n=size[g],s=start[g],nr=3*(rp[g+1]-rp[g]);const double*A=val+vp[g];
  double v=0.;for(int k=lane;k<nr;k+=32){int target=3*rows[rp[g]+k/3]+k%3;v+=A[(long long)k*n+r]*x[(long long)target*width+j];}
  v=warp_sum(v);if(!lane)rhs[(long long)(s+r)*width+j]=x[(long long)(s+r)*width+j]-v;
 }
}
'''

def prepare_supernodes(host,max_nodes=256):
    start_time=time.perf_counter();ptr,rows,blocks=host.ptr,host.rows,host.blocks
    n=len(ptr)-1;groups=[];i=0
    while i<n:
        end=i+1
        while end<n and end-i<max_nodes:
            a=rows[ptr[end-1]+1:ptr[end]];b=rows[ptr[end]:ptr[end+1]]
            if not np.array_equal(a,b):break
            end+=1
        groups.append((i,end));i=end
    owner=np.empty(n,np.int32)
    for g,(lo,hi) in enumerate(groups):owner[lo:hi]=g
    starts=[];sizes=[];rptr=[0];vptr=[0];iptr=[0];allrows=[];values=[];inverses=[]
    levels=np.zeros(len(groups),np.int32)
    for g,(lo,hi) in enumerate(groups):
        outside=rows[ptr[lo]:ptr[lo+1]];outside=outside[outside>=hi]
        d=3*(hi-lo);D=np.zeros((d,d));E=np.empty((3*len(outside),d))
        for col in range(lo,hi):
            rr=rows[ptr[col]:ptr[col+1]];bb=blocks[ptr[col]:ptr[col+1]];inside=rr<hi
            cc=3*(col-lo)
            for a in range(3):
                for b in range(3):D[3*(rr[inside]-lo)+a,cc+b]=bb[inside,a,b]
            ext=bb[~inside]
            if not np.array_equal(rr[~inside],outside):raise ValueError('Invalid supernode panel')
            E[:,cc:cc+3]=ext.reshape(-1,3)
        inv=la.solve_triangular(D,np.eye(d),lower=True,unit_diagonal=True,check_finite=False)
        starts.append(3*lo);sizes.append(d);allrows.append(outside);rptr.append(rptr[-1]+len(outside))
        values.append(E.ravel());vptr.append(vptr[-1]+E.size)
        inverses.append(inv.ravel());iptr.append(iptr[-1]+inv.size)
        targets=np.unique(owner[outside]);levels[targets]=np.maximum(levels[targets],levels[g]+1)
    data=dict(start=np.asarray(starts,np.int32),size=np.asarray(sizes,np.int32),rp=np.asarray(rptr,np.int32),
              rows=np.concatenate(allrows).astype(np.int32),vp=np.asarray(vptr,np.int64),val=np.concatenate(values),
              ip=np.asarray(iptr,np.int64),iv=np.concatenate(inverses),level=levels)
    host.supernodes=data
    host.supernode_record=dict(groups=len(groups),levels=int(levels.max()+1),maximum_scalar_panel=max(sizes),
        preparation_s=time.perf_counter()-start_time,parameter_bytes=sum(v.nbytes for v in data.values()))
    return data

class DeviceMeshSupernode:
    def __init__(self,host,surface_modes=None):
        if not hasattr(host,'supernodes'):prepare_supernodes(host)
        data=host.supernodes;self.data={k:cp.asarray(v) for k,v in data.items() if k!='level'}
        self.levels=[cp.asarray(np.flatnonzero(data['level']==i).astype(np.int32)) for i in range(data['level'].max()+1)]
        self.tasks=[]
        for level in range(data['level'].max()+1):
            ids=np.flatnonzero(data['level']==level);ds=data['size'][ids];fs=3*np.diff(data['rp'])[ids]
            arrays=[np.repeat(ids,ds),np.concatenate([np.arange(n) for n in ds]),
                    np.repeat(ids,fs),np.concatenate([np.arange(n) for n in fs])]
            self.tasks.append(tuple(cp.asarray(a,dtype=cp.int32) for a in arrays))
        self.P=cp.asarray(host.P) if surface_modes is None else surface_modes
        self.Kinv=cp.asarray(host.Kinv);self.D=cp.asarray(host.diagonal);self.surface=cp.asarray(host.surface)
        self.n=len(self.surface);self.nr=len(self.D);self.work={}
        module=cp.RawModule(code=CODE);self.k={n:module.get_function(n) for n in
            ('diagonal_panel','forward_panel','backward_panel','level_diagonal','level_forward','level_backward','scatter_input','gather_output')}
        self.parameter_bytes=sum(a.nbytes for a in [*self.data.values(),*self.levels,self.P,self.Kinv,self.D,self.surface,
                                                  *(a for task in self.tasks for a in task)])
        self.record=host.supernode_record;self.reserve(1)

    def reserve(self,width):
        if width not in self.work:
            # Capacity growth occurs only outside a running/captured graph.
            # Old working-set widths are superseded, not retained as caches.
            if width>1:
                cp.cuda.get_current_stream().synchronize()
                for old in list(self.work):
                    if old>1:del self.work[old]
            self.work[width]=dict(rhs=cp.zeros((self.nr,width)),x=cp.zeros((self.nr,width)),f=cp.zeros((self.n,width)),
                out=cp.zeros((self.n,width)),modal=cp.zeros((self.P.shape[1],width)),
                correction=cp.zeros((self.P.shape[1],width)),default_meta=cp.asarray([0,0,width,0,0,0,0],dtype=cp.int32))
        return self.work[width]

    def apply(self,f,modal_force=None,meta=None,modal_inverse=None):
        one=f.ndim==1;width=1 if one else f.shape[1];w=self.work[width];d=self.data
        selected_meta=meta
        if meta is None:meta=w['default_meta']
        F=f.reshape(self.n,width);w['rhs'].fill(0);w['x'].fill(0)
        self.k['scatter_input'](((self.n*width+127)//128,),(128,),(F,self.surface,w['rhs'],np.int32(self.n),np.int32(width)))
        diag=(d['start'],d['size'],d['ip'],d['iv'])
        off=(d['start'],d['size'],d['rp'],d['rows'],d['vp'],d['val'])
        for dg,dr,fg,fr in self.tasks:
            self.k['level_diagonal']((min(256,max(1,(len(dg)*width+3)//4)),),(128,),
                (dg,dr,np.int32(len(dg)),*diag,w['rhs'],w['x'],np.int32(width),meta,np.int32(0)))
            self.k['level_forward']((min(256,max(1,(len(fg)*width+127)//128)),),(128,),
                (fg,fr,np.int32(len(fg)),*off,w['x'],w['rhs'],np.int32(width),meta))
        w['x']/=self.D[:,None]
        for dg,dr,fg,fr in reversed(self.tasks):
            self.k['level_backward']((min(256,max(1,(len(dg)*width+3)//4)),),(128,),
                (dg,dr,np.int32(len(dg)),*off,w['x'],w['rhs'],np.int32(width),meta))
            self.k['level_diagonal']((min(256,max(1,(len(dg)*width+3)//4)),),(128,),
                (dg,dr,np.int32(len(dg)),*diag,w['rhs'],w['x'],np.int32(width),meta,np.int32(1)))
        if modal_force is None:matmul(self.P.T,F,w['modal'],selected_meta)
        else:w['modal'][:]=modal_force.reshape(-1,width)
        inverse=self.Kinv if modal_inverse is None else modal_inverse
        matmul(inverse,w['modal'],w['correction'],selected_meta);matmul(self.P,w['correction'],w['out'],selected_meta)
        self.k['gather_output'](((self.n*width+127)//128,),(128,),(w['x'],self.surface,w['out'],np.int32(self.n),np.int32(width)))
        return w['out'][:,0] if one else w['out']

    def storage(self):
        workspace=sum(v.nbytes for w in self.work.values() for v in w.values())
        return dict(operator_parameter_bytes=self.parameter_bytes,query_workspace_bytes=workspace,sparse_analysis_bytes=0,
            stored_bytes=self.parameter_bytes+workspace,dense_surface_or_region_matrix_bytes=0,widths=list(self.work),supernodes=self.record)
