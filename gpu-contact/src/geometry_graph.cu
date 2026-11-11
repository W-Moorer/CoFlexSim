extern "C" __global__ void pack_pairs(int*out,const int*counts,const int*offsets,int faces,int slots,int capacity,
 const int*fs,const int*fm,const double*x,int soff,int moff,int*owners,int*mids,int*nodes,double*S,double*M,int*overflow){
 int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=faces)return;int n=min(counts[f],slots),base=offsets[f];
 if(base+n>capacity){atomicExch(overflow,3);return;}
 for(int j=1;j<n;j++){int val=out[f*slots+j],k=j-1;while(k>=0&&out[f*slots+k]>val){out[f*slots+k+1]=out[f*slots+k];k--;}out[f*slots+k+1]=val;}
 for(int j=0;j<n;j++){int i=base+j,m=out[f*slots+j];owners[i]=f;mids[i]=m;
  for(int k=0;k<3;k++){int a=fs[3*f+k]+soff,b=fm[3*m+k]+moff;nodes[6*i+k]=a;nodes[6*i+3+k]=b;
   for(int d=0;d<3;d++){S[9*i+3*k+d]=x[3*a+d];M[9*i+3*k+d]=x[3*b+d];}}}
}
extern "C" __global__ void dense_support(const double*area,const double*numer,const int*prefix,int n,int*lookup,
 double*gaps,long long*keys,long long base){
 int node=blockIdx.x*blockDim.x+threadIdx.x;if(node>=n)return;int row=area[node]>1e-24?prefix[node]-1:-1;
 lookup[node]=row;if(row>=0){gaps[row]=numer[node]/area[node];keys[row]=base+node;}
}
extern "C" __global__ void geometry_report(const int*prefix,int n,const int*voff,int cap,const int*ptr,
 const int*poff,int nf,const int*qerror,const int*aerror,const int*samples,int*report,int derivatives){
 if(threadIdx.x)return;int nw=prefix[n-1],m=nw+voff[cap],overlap=0;
 for(int i=0;i<cap;i++)overlap+=samples[2*i]>0;
 report[0]=poff[nf];report[1]=nw;report[2]=m;report[3]=derivatives?ptr[m]:0;
 report[4]=qerror[0];report[5]=aerror[0];report[6]=overlap;
}
