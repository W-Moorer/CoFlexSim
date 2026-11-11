"""CUDA contact actions preserving additive element entries without sorting.

Duplicate entries are intentional FE contributions. These kernels never ask a
generic sparse library to canonicalize them or construct column submatrices.
"""
from types import SimpleNamespace
import numpy as np
import cupy as cp

CODE=r'''
extern "C" __global__ void row_action(const int*p,const int*c,const double*a,const double*x,double*y,int n){
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=n)return;double v=0.;
 for(int e=p[r];e<p[r+1];e++)v+=a[e]*x[c[e]];y[r]=v;
}
extern "C" __global__ void adjoint_action(const int*p,const int*c,const double*a,const double*x,double*y,int n){
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=n||x[r]==0.)return;
 for(int e=p[r];e<p[r+1];e++)if(a[e]!=0.)atomicAdd(y+c[e],a[e]*x[r]);
}
extern "C" __global__ void body_rows(const int*c,const double*a,int*bc,double*ba,int groups,int offset,int n,const double*R){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=groups)return;int col=c[3*j]-offset;
 for(int k=0;k<3;k++){
  bc[3*j+k]=(col>=0&&col<n)?col+k:k;
  ba[3*j+k]=(col>=0&&col<n)?a[3*j]*R[k]+a[3*j+1]*R[3+k]+a[3*j+2]*R[6+k]:0.;
 }
}
extern "C" __global__ void reference_rows(const int*p,const int*c,const double*a,const int*bc,const double*ba,
 const double*tangent,const double*pu,const double*sigma,int offset,int n,int rows,double*jr,double*v,double*rank){
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=rows)return;double d=0.,u=0.;
 for(int e=p[r];e<p[r+1];e++){int k=c[e]-offset;if(k>=0&&k<n)d+=a[e]*tangent[k];u+=ba[e]*pu[bc[e]];}
 jr[r]=d;v[r]=d-u;rank[r]=(d-u)/sqrt(sigma[0]);
}
'''

class ContactActions:
    def __init__(self):
        module=cp.RawModule(code=CODE)
        self.k={name:module.get_function(name) for name in ('row_action','adjoint_action','body_rows','reference_rows')}

    def action(self,A,x):
        n=A.shape[0];y=cp.empty(n)
        if n:self.k['row_action'](((n+127)//128,),(128,),(A.indptr,A.indices,A.data,x,y,np.int32(n)))
        return y

    def adjoint(self,A,x):
        n=A.shape[0];y=cp.zeros(A.shape[1])
        if n:self.k['adjoint_action'](((n+127)//128,),(128,),(A.indptr,A.indices,A.data,x,y,np.int32(n)))
        return y

    def body(self,W,R,tangent,pu,sigma,offset,size):
        if len(W.data)%3:raise RuntimeError('Contact CSR must preserve complete xyz triplets')
        groups=len(W.data)//3;rows=W.shape[0];data=cp.empty_like(W.data);cols=cp.empty_like(W.indices)
        if groups:self.k['body_rows'](((groups+127)//128,),(128,),
            (W.indices,W.data,cols,data,np.int32(groups),np.int32(offset),np.int32(size),R))
        B=SimpleNamespace(data=data,indices=cols,indptr=W.indptr,shape=(rows,size))
        jr=cp.zeros(rows);v=cp.zeros(rows);rank=cp.zeros(rows)
        if tangent is not None and rows:
            self.k['reference_rows'](((rows+127)//128,),(128,),
                (W.indptr,W.indices,W.data,cols,data,tangent,pu,sigma,np.int32(offset),np.int32(size),np.int32(rows),jr,v,rank))
        return B,jr,v,rank
