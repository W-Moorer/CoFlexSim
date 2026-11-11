// Host dispatches one graph per working-set expansion; numerical pivots and
// all-candidate feasibility remain on CUDA. meta: m, seeds, count, cached, done,
// error, expansions. Padding variables have an exact zero solution.
extern "C" __global__ void set_start(const double*b,int*meta,long long*ids,double*weights,int cap){
 if(threadIdx.x)return;int m=meta[0],k=meta[1],j=0;for(int i=1;i<m;i++)if(b[i]<b[j])j=i;
 bool found=false;for(int i=0;i<k;i++)found=found||ids[i]==j;
 if(!found){if(k==cap){meta[5]=5;return;}ids[k]=j;weights[k++]=0.;}
 meta[2]=k;meta[3]=0;meta[4]=0;meta[5]=0;meta[6]=0;
}
extern "C" __global__ void new_response(const double*C,const int*ptr,const int*col,const double*val,
 const long long*ids,const int*meta,double*T,int p,int cap){
 int idx=blockIdx.x*blockDim.x+threadIdx.x,k=meta[2]-meta[3];if(idx>=p*k)return;int i=idx/k,j=meta[3]+idx%k;
 int row=ids[j];double v=0.;
 for(int e=ptr[row];e<ptr[row+1];e++)if(val[e]!=0.)v+=C[(long long)i*p+col[e]]*val[e];T[i*cap+j]=v;
}
extern "C" __global__ void new_columns(const int*ptr,const int*col,const double*val,const double*v,
 const long long*ids,const int*meta,const double*T,double*W,int cap,int add){
 int idx=blockIdx.x*blockDim.x+threadIdx.x,k=meta[2]-meta[3];if(idx>=meta[0]*k)return;int i=idx/k,j=meta[3]+idx%k;
 double s=v[i]*v[ids[j]];for(int e=ptr[i];e<ptr[i+1];e++)if(val[e]!=0.)s+=val[e]*T[col[e]*cap+j];
 if(add)W[i*cap+j]+=s;else W[i*cap+j]=s;
}
extern "C" __global__ void gather_block(const double*W,const double*b,const long long*ids,const int*meta,
 double*C,double*rhs,double*weights,int cap){
 int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx>=cap*cap)return;int i=idx/cap,j=idx%cap,k=meta[2];
 C[idx]=(i<k&&j<k)?W[ids[i]*cap+j]:(i==j?1.:0.);
 if(!j){rhs[i]=i<k?b[ids[i]]:0.;if(i>=k)weights[i]=0.;}
}
extern "C" __global__ void check_all(const double*W,const double*b,const double*weights,const long long*ids,
 const int*meta,double*partial,int*worst,int cap){
 int r=blockIdx.x*blockDim.x+threadIdx.x;double eligible=1e300,ae=0.;int id=-1;
 if(r<meta[0]){double gap=b[r];bool active=false;for(int j=0;j<meta[2];j++){gap+=W[r*cap+j]*weights[j];active=active||ids[j]==r;}
  if(active)ae=fmax(-gap,0.);else {eligible=gap;id=r;}}
 __shared__ double s[256],err[256];__shared__ int which[256];int lane=threadIdx.x;s[lane]=eligible;err[lane]=ae;which[lane]=id;__syncthreads();
 for(int stride=128;stride;stride/=2){if(lane<stride){if(s[lane+stride]<s[lane]||(s[lane+stride]==s[lane]&&which[lane+stride]<which[lane])){s[lane]=s[lane+stride];which[lane]=which[lane+stride];}err[lane]=fmax(err[lane],err[lane+stride]);}__syncthreads();}
 if(!lane){partial[2*blockIdx.x]=s[0];partial[2*blockIdx.x+1]=err[0];worst[blockIdx.x]=which[0];}
}
extern "C" __global__ void next_column(const double*partial,const int*worst,const int*inner,int*meta,
 long long*ids,double*weights,int blocks,int cap,double tolerance){
 if(threadIdx.x)return;meta[6]++;if(inner[0]){meta[5]=inner[0];return;}
 double low=1e300,ae=0.;int j=-1;for(int i=0;i<blocks;i++){
  if(partial[2*i]<low||(partial[2*i]==low&&worst[i]<j)){low=partial[2*i];j=worst[i];}ae=fmax(ae,partial[2*i+1]);}
 if(low>=-10*tolerance){if(ae>100*tolerance)meta[5]=6;else meta[4]=1;return;}
 int k=meta[2];if(k>=cap){meta[5]=5;return;}meta[3]=k;ids[k]=j;weights[k]=0.;meta[2]=k+1;
}
extern "C" __global__ void scatter_result(const long long*ids,const double*weights,const int*meta,double*lambda,int cap){
 int i=threadIdx.x;if(i<meta[2])lambda[ids[i]]=weights[i];
}
