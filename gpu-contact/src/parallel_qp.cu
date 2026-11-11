// Parallel local two-row contact patches followed by an exact coupled active-set
// correction. One cooperative CTA owns a bounded working matrix; rows, patch
// predictions and Cholesky column updates use different lanes. Scalar pivots are
// device decisions. No diagonal regularization is added to the physical operator.
__device__ double reduce_max(double value, double* scratch) {
 int t=threadIdx.x;scratch[t]=value;__syncthreads();
 for(int s=blockDim.x/2;s;s/=2){if(t<s)scratch[t]=fmax(scratch[t],scratch[t+s]);__syncthreads();}
 return scratch[0];
}
__device__ double coeff(const double*C,const double*scale,int i,int j,int n){
 return .5*(C[i*n+j]+C[j*n+i])/(scale[i]*scale[j]);
}
extern "C" __global__ void parallel_qp(const double*C,const double*b,
 const double*initial,double*result,int*report,const int*meta,double*L,
 int n,double tol,int maxit,int local_sweeps){
 if(meta[4]||meta[5])return;
 extern __shared__ double storage[];
 double*scale=storage;double*x=scale+n;double*z=x+n;double*f=z+n;
 double*gap=f+n;double*work=gap+n;double*scratch=work+n;
 int*active=(int*)(scratch+128);int*ids=active+n;
 __shared__ int k,worst,failed,done,rank,chosen;
 __shared__ double pivot,alpha;
 int t=threadIdx.x;
 if(!t){report[0]=0;report[1]=0;report[2]=0;report[3]=0;failed=0;done=0;}
 for(int i=t;i<n;i+=blockDim.x){
  scale[i]=sqrt(fmax(C[i*n+i],1e-30));f[i]=b[i]/scale[i];x[i]=fmax(initial[i],0.)*scale[i];
 }
 __syncthreads();
 // Additive nonoverlapping two-row blocks. Cross-patch loads always use the
 // same previous iterate; damping bounds the predictor. Final accuracy never
 // depends on this approximation: the global corrector checks the original QP.
 for(int sweep=0;sweep<local_sweeps;sweep++){
  for(int pair=t;pair<(n+1)/2;pair+=blockDim.x){
   int a=2*pair,c=a+1;double fa=f[a],fc=c<n?f[c]:0.;
   for(int j=0;j<n;j++)if(j!=a&&j!=c){fa+=coeff(C,scale,a,j,n)*x[j];if(c<n)fc+=coeff(C,scale,c,j,n)*x[j];}
   double za=fmax(-fa,0.),zc=0.;
   if(c<n){
    double r=coeff(C,scale,a,c,n),det=1-r*r;
    if(det>1e-12){
     double u=(-fa+r*fc)/det,v=(-fc+r*fa)/det;
     if(u>=0.&&v>=0.){za=u;zc=v;}
     else if(za>0.&&fc+r*za>=0.)zc=0.;
     else{zc=fmax(-fc,0.);za=0.;}
    }else{za=fmax(x[a]-(fa+x[a]+r*x[c]),0.);zc=fmax(x[c]-(fc+x[c]+r*x[a]),0.);}
   }
   z[a]=.5*x[a]+.5*za;if(c<n)z[c]=.5*x[c]+.5*zc;
  }
  __syncthreads();for(int i=t;i<n;i+=blockDim.x)x[i]=z[i];__syncthreads();
 }
 for(int i=t;i<n;i+=blockDim.x)active[i]=x[i]>0.;__syncthreads();
 for(int it=0;it<maxit;it++){
  for(int i=t;i<n;i+=blockDim.x){double v=f[i];for(int j=0;j<n;j++)v+=coeff(C,scale,i,j,n)*x[j];gap[i]=v*scale[i];}
  __syncthreads();
  if(!t){
   report[1]=it+1;worst=-1;double minimum=-tol,ae=0.;
   for(int i=0;i<n;i++){if(active[i])ae=fmax(ae,fabs(gap[i]));else if(gap[i]<minimum){minimum=gap[i];worst=i;}}
   done=(worst<0&&ae<=10*tol);if(worst>=0)active[worst]=1;
  }
  __syncthreads();
  if(done){
   for(int i=t;i<n;i+=blockDim.x)result[i]=x[i]/scale[i];
   if(!t){int count=0;for(int i=0;i<n;i++)count+=x[i]/scale[i]>1e-10;report[2]=count;report[3]=local_sweeps;}
   return;
  }
  bool inner_done=false;
  for(int inner=0;inner<2*n+2;inner++){
   if(!t){k=0;for(int i=0;i<n;i++)if(active[i])ids[k++]=i;}
   for(int i=t;i<n;i+=blockDim.x)z[i]=0.;__syncthreads();
   // Complete-pivot Cholesky. A dependent row yields a null direction; move
   // along it to a nonnegative boundary, rather than regularizing compliance.
   if(!t){rank=k;failed=0;}__syncthreads();
   for(int col=0;col<k;col++){
    for(int row=col+t;row<k;row+=blockDim.x){
     double v=coeff(C,scale,ids[row],ids[row],n);for(int j=0;j<col;j++)v-=L[row*n+j]*L[row*n+j];work[row]=v;
    }
    __syncthreads();
    if(!t){chosen=col;for(int row=col+1;row<k;row++)if(work[row]>work[chosen])chosen=row;
     if(!(work[chosen]>1e-13)){rank=col;failed=1;}else{int temp=ids[col];ids[col]=ids[chosen];ids[chosen]=temp;}}
    __syncthreads();if(failed)break;
    for(int j=t;j<col;j+=blockDim.x){double temp=L[col*n+j];L[col*n+j]=L[chosen*n+j];L[chosen*n+j]=temp;}__syncthreads();
    for(int row=col+t;row<k;row+=blockDim.x){
     double v=coeff(C,scale,ids[row],ids[col],n);
     for(int j=0;j<col;j++)v-=L[row*n+j]*L[col*n+j];work[row]=v;
    }
    __syncthreads();if(!t){pivot=work[col];if(!(pivot>0.)){failed=1;rank=col;}else L[col*n+col]=sqrt(pivot);}
    __syncthreads();if(failed)break;
    for(int row=col+1+t;row<k;row+=blockDim.x)L[row*n+col]=work[row]/L[col*n+col];__syncthreads();
   }
   if(failed){
    // v_dep=-1, v_ind=A_ind^-1 A_ind,dep is an exact null direction
    // up to the rank tolerance. Choose its sign to avoid increasing objective.
    for(int i=t;i<rank;i+=blockDim.x)work[i]=coeff(C,scale,ids[i],ids[rank],n);__syncthreads();
    for(int col=0;col<rank;col++){
     if(!t)work[col]/=L[col*n+col];__syncthreads();
     for(int row=col+1+t;row<rank;row+=blockDim.x)work[row]-=L[row*n+col]*work[col];__syncthreads();
    }
    for(int col=rank-1;col>=0;col--){
     if(!t)work[col]/=L[col*n+col];__syncthreads();
     for(int row=t;row<col;row+=blockDim.x)work[row]-=L[col*n+row]*work[col];__syncthreads();
    }
    for(int i=t;i<rank;i+=blockDim.x)z[ids[i]]=work[i];
    if(!t)z[ids[rank]]=-1.;__syncthreads();
    if(!t){
     double derivative=0.;for(int i=0;i<n;i++)derivative+=f[i]*z[i];
     pivot=derivative>0.?-1.:1.;alpha=1e300;chosen=-1;
     for(int i=0;i<n;i++)if(pivot*z[i]<-1e-14){double ratio=x[i]/(-pivot*z[i]);if(ratio<alpha){alpha=ratio;chosen=i;}}
     if(chosen<0)report[0]=7;
    }
    __syncthreads();if(chosen<0)return;
    for(int i=t;i<n;i+=blockDim.x)x[i]=fmax(x[i]+alpha*pivot*z[i],0.);
    __syncthreads();if(!t){active[chosen]=0;x[chosen]=0.;report[3]++;}__syncthreads();
    continue;
   }
   for(int i=t;i<k;i+=blockDim.x)work[i]=-f[ids[i]];__syncthreads();
   for(int col=0;col<k;col++){
    if(!t)work[col]/=L[col*n+col];__syncthreads();
    for(int row=col+1+t;row<k;row+=blockDim.x)work[row]-=L[row*n+col]*work[col];__syncthreads();
   }
   for(int col=k-1;col>=0;col--){
    if(!t)work[col]/=L[col*n+col];__syncthreads();
    for(int row=t;row<col;row+=blockDim.x)work[row]-=L[col*n+row]*work[col];__syncthreads();
   }
   for(int i=t;i<k;i+=blockDim.x)z[ids[i]]=work[i];__syncthreads();
   double error=0.;
   for(int refinement=0;refinement<3;refinement++){
    double defect=0.;
    for(int row=t;row<k;row+=blockDim.x){int i=ids[row];double v=f[i];for(int j=0;j<k;j++)v+=coeff(C,scale,i,ids[j],n)*z[ids[j]];work[row]=-v;defect=fmax(defect,fabs(v*scale[i]));}
    error=reduce_max(defect,scratch);if(error<=tol||refinement==2)break;
    for(int col=0;col<k;col++){
     if(!t)work[col]/=L[col*n+col];__syncthreads();
     for(int row=col+1+t;row<k;row+=blockDim.x)work[row]-=L[row*n+col]*work[col];__syncthreads();
    }
    for(int col=k-1;col>=0;col--){
     if(!t)work[col]/=L[col*n+col];__syncthreads();
     for(int row=t;row<col;row+=blockDim.x)work[row]-=L[col*n+row]*work[col];__syncthreads();
    }
    for(int i=t;i<k;i+=blockDim.x)z[ids[i]]+=work[i];__syncthreads();
   }
   if(error>tol){if(!t)report[0]=2;return;}
   if(!t){
    alpha=1.;done=1;
    for(int i=0;i<n;i++)if(active[i]&&z[i]<=0.){done=0;alpha=fmin(alpha,x[i]/fmax(x[i]-z[i],1e-300));}
   }
   __syncthreads();
   if(done){for(int i=t;i<n;i+=blockDim.x)x[i]=z[i];__syncthreads();inner_done=true;break;}
   double maximum=1.;for(int i=t;i<n;i+=blockDim.x){x[i]+=alpha*(z[i]-x[i]);maximum=fmax(maximum,x[i]);}
   maximum=reduce_max(maximum,scratch);
   for(int i=t;i<n;i+=blockDim.x)if(active[i]&&x[i]<=1e-13*maximum){active[i]=0;x[i]=0.;}__syncthreads();
  }
  if(!inner_done){if(!t)report[0]=3;return;}
 }
 if(!t)report[0]=4;
}
