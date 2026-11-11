extern "C" __global__ void scatter_factor(const double* src, double* dst,
    const long long* slots, int count, int rank, int start) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<(long long)count*rank) dst[slots[start+i/rank]*rank+i%rank]=src[i];
}
extern "C" __global__ void region_mark_pairs(const int*owner,const int*master,const int*sf,const int*mf,
 int*mask,int n,int slo,int mlo){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 int s=owner[i],m=master[i];if(s<0||m<0)return;
 for(int k=0;k<3;k++){atomicExch(mask+slo+sf[3*s+k],1);atomicExch(mask+mlo+mf[3*m+k],1);}
}
extern "C" __global__ void region_mark_load(const int*ptr,const int*col,const double*val,
 int*mask,int cap,int rows){
 int end=min(cap,ptr[rows]);
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<end;i+=gridDim.x*blockDim.x)
  if(val[i]!=0.)atomicExch(mask+col[i]/3,1);
}
extern "C" __global__ void region_mark_ready(const int*ptr,const int*nodes,const int*lookup,int*out,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int ready=1;
 for(int e=ptr[i];e<ptr[i+1]&&ready;e++){int v=3*nodes[e];ready=lookup[v]>=0&&lookup[v+1]>=0&&lookup[v+2]>=0;}
 out[i]=ready;
}
extern "C" __global__ void region_modal_rows(const double*P,const double*A,double*D,
 const long long*ids,const long long*slots,int n,int m){
 int k=blockIdx.x*blockDim.x+threadIdx.x,i=blockIdx.y;if(k>=m||i>=n)return;
 double sum=0.;for(int j=0;j<m;j++)sum+=P[ids[i]*(long long)m+j]*A[k*(long long)m+j];
 D[slots[i]*(long long)m+k]=sum;
}
extern "C" __global__ void gather_factor(const double* src, double* dst,
    const long long* slots, int count, int rank, int start) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<(long long)count*rank) dst[i]=src[slots[start+i/rank]*rank+i%rank];
}
extern "C" __global__ void scatter_response(const double* src, double* dst,
    const long long* rows, const long long* cols, int nr, int nc,
    int cap, int start, int transpose) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=(long long)nr*nc)return;
    long long r=rows[i/nc], c=cols[start+i%nc];
    dst[transpose ? c+r*cap : r+c*cap]=src[i];
}
