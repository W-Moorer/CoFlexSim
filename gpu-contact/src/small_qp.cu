// Scaled, unregularized primal working-set QP. One CUDA block owns a small
// coupled system; numerical work and pivot decisions stay on the device.
// A rank-deficient/ill-conditioned block requests the CUDA eigensolver fallback.
extern "C" __global__ void small_qp(const double* C,const double* b,
 const double* initial,double* result,int* report,int n,double tol,int maxit) {
 if(threadIdx.x || blockIdx.x)return;
 extern __shared__ double mem[];
 double* L=mem;double* scale=L+n*n;double* x=scale+n;double* z=x+n;
 double* work=z+n;double* gap=work+n;double* solution=gap+n;
 int* active=(int*)(solution+n);int* ids=active+n;
 report[0]=0;report[1]=0;report[2]=0;
 for(int i=0;i<n;i++){
  scale[i]=sqrt(fmax(C[i*n+i],1e-30));x[i]=fmax(initial[i],0.)*scale[i];active[i]=x[i]>0.;
 }
 for(int it=0;it<maxit;it++){
  report[1]=it+1;int worst=-1;double minimum=0.,ae=0.;
  for(int i=0;i<n;i++){
   double w=b[i];for(int j=0;j<n;j++)w+=.5*(C[i*n+j]+C[j*n+i])*x[j]/scale[j];
   gap[i]=w;if(active[i])ae=fmax(ae,fabs(w));else if(w<minimum){minimum=w;worst=i;}
  }
  if(minimum>=-tol && ae<=10*tol){
   int count=0;for(int i=0;i<n;i++){result[i]=x[i]/scale[i];if(result[i]>1e-10)count++;}
   report[2]=count;return;
  }
  if(minimum<-tol)active[worst]=1;
  bool inner_done=false;
  for(int inner=0;inner<2*n+2;inner++){
   int k=0;for(int i=0;i<n;i++){z[i]=0.;if(active[i])ids[k++]=i;}
   for(int i=0;i<k;i++)for(int j=0;j<=i;j++){
    int a=ids[i],c=ids[j];double sum=.5*(C[a*n+c]+C[c*n+a])/(scale[a]*scale[c]);
    for(int p=0;p<j;p++)sum-=L[i*n+p]*L[j*n+p];
    if(i==j){if(!(sum>1e-14)){report[0]=1;return;}L[i*n+j]=sqrt(sum);}
    else L[i*n+j]=sum/L[j*n+j];
   }
   for(int i=0;i<k;i++){
    double v=-b[ids[i]]/scale[ids[i]];for(int j=0;j<i;j++)v-=L[i*n+j]*work[j];
    work[i]=v/L[i*n+i];
   }
   for(int i=k-1;i>=0;i--){
    double v=work[i];for(int j=i+1;j<k;j++)v-=L[j*n+i]*solution[j];
    solution[i]=v/L[i*n+i];z[ids[i]]=solution[i];
   }
   for(int i=0;i<k;i++){
    int a=ids[i];double defect=b[a];
    for(int j=0;j<k;j++){int c=ids[j];defect+=.5*(C[a*n+c]+C[c*n+a])*z[c]/scale[c];}
    if(!isfinite(defect)||fabs(defect)>tol){report[0]=2;return;}
   }
   bool negative=false;double alpha=1.;
   for(int i=0;i<n;i++)if(active[i]&&z[i]<=0.){
    negative=true;alpha=fmin(alpha,x[i]/fmax(x[i]-z[i],1e-300));
   }
   if(!negative){for(int i=0;i<n;i++)x[i]=z[i];inner_done=true;break;}
   double maximum=1.;for(int i=0;i<n;i++){x[i]+=alpha*(z[i]-x[i]);maximum=fmax(maximum,x[i]);}
   for(int i=0;i<n;i++)if(active[i]&&x[i]<=1e-13*maximum){active[i]=0;x[i]=0.;}
  }
  if(!inner_done){report[0]=3;return;}
 }
 report[0]=4;
}
