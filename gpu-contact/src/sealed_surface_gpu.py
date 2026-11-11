"""Fused CUDA application of a sealed whole-surface response operator.

Exact equal-load reuse is device controlled. Metadata visits only blocks with
nonzero applied loads; matrix blocks and cross-region interactions stay intact.
Geometry is not frozen. Full geometry coverage checks remain in the simulator.
"""
import numpy as np
import cupy as cp

CODE=r'''
extern "C" __global__ void changed_force(const double*f,const double*old,int*changed,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&f[i]!=old[i])atomicExch(changed,1);
}
extern "C" __global__ void static_load(const double*P,const double*f,double*out,int n,int nm){
 int k=blockIdx.x,t=threadIdx.x;double s=0.;
 for(int i=t;i<n;i+=blockDim.x)if(f[i]!=0.)s+=P[(long long)i*nm+k]*f[i];
 __shared__ double z[128];z[t]=s;__syncthreads();
 for(int d=64;d;d/=2){if(t<d)z[t]+=z[t+d];__syncthreads();}if(!t)out[k]=z[0];
}
extern "C" __global__ void static_coefficients(const double*A,const double*f,double*out,int nm){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nm)return;double s=0.;
 for(int k=0;k<nm;k++)s+=A[(long long)i*nm+k]*f[k];out[i]=s;
}
// part: row-index offset, col-index offset, nr, nc, rank (-1=dense),
//       first value offset, second value offset, transpose, projection offset.
extern "C" __global__ void prepare_blocks(const long long*parts,const int*indices,
 const double*values,const double*f,double*projection,int*active,const int*changed){
 if(!changed[0])return;int b=blockIdx.x,t=threadIdx.x;const long long*m=parts+9*b;
 const int*cols=indices+m[1];int nc=m[3],rank=m[4];__shared__ int touched;
 if(!t)touched=0;__syncthreads();
 for(int j=t;j<nc;j+=blockDim.x)if(f[cols[j]]!=0.)atomicExch(&touched,1);
 __syncthreads();if(!t)active[b]=touched;if(!touched||rank<=0)return;
 const double*V=values+m[6];
 for(int k=t;k<rank;k+=blockDim.x){double s=0.;
  for(int j=0;j<nc;j++)if(f[cols[j]]!=0.)s+=V[(long long)j*rank+k]*f[cols[j]];
  projection[m[8]+k]=s;
 }
}
extern "C" __global__ void recover_rows(const long long*parts,const int*indices,
 const double*values,const double*projection,const int*active,const int*ptr,
 const int*part_ids,const int*local_rows,const double*f,double*out,
 const int*changed,int n,long long*counts,const double*modes,const double*amplitudes,int nm){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 if(!i)counts[!changed[0]]+=1LL;if(!changed[0])return;
 double sum=0.;for(int e=ptr[i];e<ptr[i+1];e++){
  int b=part_ids[e];if(!active[b])continue;const long long*m=parts+9*b;
  int r=local_rows[e],nc=m[3],rank=m[4];const double*A=values+m[5];
  if(rank<0){const int*cols=indices+m[1];
   for(int j=0;j<nc;j++){double z=f[cols[j]];if(z!=0.)sum+=A[m[7]?(long long)j*m[2]+r:(long long)r*nc+j]*z;}
  }else for(int k=0;k<rank;k++)sum+=A[(long long)r*rank+k]*projection[m[8]+k];
 }for(int k=0;k<nm;k++)sum-=modes[(long long)i*nm+k]*amplitudes[k];out[i]=sum;
}
'''


class DeviceSealedOperator:
    def __init__(self, source, surface_modes=None):
        if not getattr(source,'sealed',False) or not source.statistics['full_surface_coverage']:
            raise ValueError('GPU online execution requires a sealed whole-surface operator')
        module=cp.RawModule(code=CODE)
        self.k={n:module.get_function(n) for n in ('changed_force','prepare_blocks','recover_rows','static_load','static_coefficients')}
        self.n=source.n;self.old=cp.full(self.n,cp.nan);self.value=cp.zeros(self.n)
        self.changed=cp.zeros(1,dtype=cp.int32);self.counts=cp.zeros(2,dtype=cp.int64)
        self.modal_count=0;self.modes=cp.empty((0,0));self.amplitudes=cp.empty(0);self.static_inverse=cp.empty((0,0))
        owned_modes=False
        if hasattr(source,'static_inverse'):
            self.modal_count=source.modes.shape[1]
            self.modes=cp.ascontiguousarray(cp.asarray(source.modes)) if surface_modes is None else surface_modes
            if self.modes.shape!=source.modes.shape:raise ValueError('Bound modal basis has wrong shape')
            if not self.modes.flags.c_contiguous:raise ValueError('Surface modes must have contiguous modal rows')
            self.static_inverse=cp.ascontiguousarray(cp.asarray(source.static_inverse));owned_modes=surface_modes is None
        self.generalized=cp.empty(self.modal_count);self.amplitudes=cp.empty(self.modal_count)
        values=[];indices=[];parts=[];incidence=[[] for _ in range(self.n)]
        nv=ni=nt=0
        def add_index(a):
            nonlocal ni
            offset=ni;indices.append(np.asarray(a,dtype=np.int32));ni+=len(a);return offset
        def add_values(a):
            nonlocal nv
            offset=nv;values.append(np.asarray(a).ravel());nv+=a.size;return offset
        def add_part(rows,cols,ri,ci,rank,first,second,trans):
            nonlocal nt
            part=len(parts);parts.append((ri,ci,len(rows),len(cols),rank,first,second,trans,nt))
            nt+=max(rank,0)
            for j,r in enumerate(rows):incidence[int(r)].append((part,j))
        for b in source.blocks:
            ri,ci=add_index(b.rows),add_index(b.cols)
            if b.dense is not None:
                a=add_values(b.dense)
                add_part(b.rows,b.cols,ri,ci,-1,a,0,0)
                if not b.diagonal:add_part(b.cols,b.rows,ci,ri,-1,a,0,1)
            else:
                u,v=add_values(b.left),add_values(b.right);rank=b.left.shape[1]
                add_part(b.rows,b.cols,ri,ci,rank,u,v,0)
                add_part(b.cols,b.rows,ci,ri,rank,v,u,0)
        ptr=np.r_[0,np.cumsum([len(v) for v in incidence])].astype(np.int32)
        adjacency=np.asarray([x for row in incidence for x in row],dtype=np.int32)
        self.parts=cp.asarray(parts,dtype=cp.int64);self.indices=cp.asarray(np.concatenate(indices))
        self.values=cp.asarray(np.concatenate(values));self.projection=cp.empty(max(1,nt))
        self.active=cp.zeros(len(parts),dtype=cp.int32);self.ptr=cp.asarray(ptr)
        self.part_ids=cp.asarray(adjacency[:,0]);self.local_rows=cp.asarray(adjacency[:,1])
        self.stored_bytes=sum(a.nbytes for a in (self.parts,self.indices,self.values,self.projection,
            self.active,self.ptr,self.part_ids,self.local_rows,self.old,self.value,self.changed,self.counts))
        self.stored_bytes+=self.static_inverse.nbytes+self.generalized.nbytes+self.amplitudes.nbytes+(self.modes.nbytes if owned_modes else 0)

    def apply(self, force, modal_force=None):
        if force.shape!=(self.n,):raise ValueError('Device surface force shape differs')
        self.changed.fill(0);grid=((self.n+127)//128,)
        self.k['changed_force'](grid,(128,),(force,self.old,self.changed,np.int32(self.n)))
        self.k['prepare_blocks']((len(self.parts),),(128,),(self.parts,self.indices,self.values,
            force,self.projection,self.active,self.changed))
        if self.modal_count:
            generalized=modal_force
            if generalized is None:
                generalized=self.generalized
                self.k['static_load']((self.modal_count,),(128,),(self.modes,force,generalized,np.int32(self.n),np.int32(self.modal_count)))
            self.k['static_coefficients'](((self.modal_count+127)//128,),(128,),
                (self.static_inverse,generalized,self.amplitudes,np.int32(self.modal_count)))
        self.k['recover_rows'](grid,(128,),(self.parts,self.indices,self.values,self.projection,self.active,
            self.ptr,self.part_ids,self.local_rows,force,self.value,self.changed,np.int32(self.n),self.counts,
            self.modes,self.amplitudes,np.int32(self.modal_count)))
        self.old[:]=force
        return self.value
