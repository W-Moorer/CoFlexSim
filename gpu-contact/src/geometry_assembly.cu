#ifndef DYNAMIC_NW
#define NW_ARG int nw
#define NW_INIT
#else
#define NW_ARG const int*nweak
#define NW_INIT int nw=*nweak;
#endif
extern "C" __global__ void support_sum(const int*owners,const int*nodes,const int*samples,const double*area,
 const double*weighted,int n,int nf,double*coverage,double*sumA,double*sumG){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 for(int kind=0;kind<2;kind++)if(samples[2*i+kind]){
  double a=0.;for(int k=0;k<3;k++)a+=area[6*i+3*kind+k];atomicAdd(coverage+kind*nf+owners[i],a);
 }
 if(samples[2*i+1])for(int k=0;k<3;k++){
  int node=nodes[6*i+k];atomicAdd(sumA+node,area[6*i+3+k]);atomicAdd(sumG+node,weighted[6*i+3+k]);
 }
}
extern "C" __global__ void support_rows(const long long*keep,int nw,const double*sumA,const double*sumG,
 int*lookup,double*gaps,long long*keys,long long keybase){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nw)return;int node=keep[i];lookup[node]=i;
 gaps[i]=sumG[node]/sumA[node];keys[i]=keybase+node;
}
extern "C" __global__ void count_entries(const int*nodes,const int*samples,const int*lookup,int n,NW_ARG,int ng,int*counts){
 NW_INIT
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n&&samples[2*i+1])for(int k=0;k<3;k++){int row=lookup[nodes[6*i+k]];if(row>=0)atomicAdd(counts+row,18);}
 if(i<ng)counts[nw+i]=18;
}
__device__ int feature(const double*x,const int*nodes,const double*p){
 double u[3],v[3],r[3],uu=0.,uv=0.,vv=0.,ru=0.,rv=0.;
 for(int a=0;a<3;a++){u[a]=x[3*nodes[1]+a]-x[3*nodes[0]+a];v[a]=x[3*nodes[2]+a]-x[3*nodes[0]+a];r[a]=p[a]-x[3*nodes[0]+a];uu+=u[a]*u[a];uv+=u[a]*v[a];vv+=v[a]*v[a];ru+=r[a]*u[a];rv+=r[a]*v[a];}
 double det=uu*vv-uv*uv;if(!(det>1e-12*uu*vv))return 0;
 double b=(vv*ru-uv*rv)/det,c=(uu*rv-uv*ru)/det,a=1-b-c;
 if(fmin(a,fmin(b,c))<-1e-7)return 0;
 return (a>1e-7?1:0)+(b>1e-7?2:0)+(c>1e-7?4:0);
}
extern "C" __global__ void assemble_rows(const int*owners,const int*mids,const int*nodes,const int*samples,
 const double*area,const double*info,const double*coverage,const double*sumA,const double*sumG,const int*lookup,
 const double*partial,const double*vjac,const double*vg,const double*vp,const int*vc,const int*voff,
 const double*x,int n,int nf,NW_ARG,int maxface,int paircode,long long guardbase,double offset,double radius,
 int*cursor,const int*ptr,int*cols,double*vals,double*gaps,long long*keys,int*error,int derivatives){
 NW_INIT
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 for(int kind=0;kind<2;kind++)if(samples[2*i+kind]&&coverage[kind*nf+owners[i]]>info[12*i+6*kind+5]*(1+1e-7))atomicExch(error,1);
 if(derivatives&&samples[2*i+1])for(int k=0;k<3;k++){
  int node=nodes[6*i+k],row=lookup[node];if(row<0)continue;double a=sumA[node],g=sumG[node]/a;
  int start=atomicAdd(cursor+row,18);
  for(int c=0;c<18;c++){cols[start+c]=3*nodes[6*i+c/3]+c%3;vals[start+c]=(partial[(i*6+3+k)*18+c]-g*partial[(i*6+k)*18+c])/a;}
 }
 for(int j=0;j<vc[i];j++){
  int row=nw+voff[i]+j;double g=vg[12*i+j];gaps[row]=g+offset;if(g < -radius)atomicExch(error,2);
  if(derivatives){
   int start=ptr[row];for(int c=0;c<18;c++){cols[start+c]=3*nodes[6*i+c/3]+c%3;vals[start+c]=vjac[(i*12+j)*18+c];}
   int a=feature(x,nodes+6*i,vp+36*i+3*j),b=feature(x,nodes+6*i+3,vp+36*i+3*j);
   long long key=guardbase+((((long long)paircode*maxface+owners[i])*maxface+mids[i])*12+j)*64+a*8+b;
   keys[row]=(a&&b&&(a!=7||b!=7))?key:-1;
  }
 }
}
