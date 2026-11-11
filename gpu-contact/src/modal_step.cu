// Fixed-size retained-coordinate dynamics only; no online volume element work.
extern "C" __global__ void modal_step(const double*K,const double*M,const double*Q,const double*D,
 const double*inv,const double*c0,const double*d0,const double*mt,
 const double*q,const double*v,const double*acc,const double*old,
 double*c,double*h,double*u,double*I,double*sigma,double*bias,
 double*nm,double*g,double*scale,double*free,int n,int o,int prismatic,int endpoint,
 double I0,double total_mass,double factor,double weight,double alpha,
 double guide_k,double guide_c,double guide_h,double target,double rest,double load,double time,double ramp){
 extern __shared__ double sm[];double*Da=sm;double*y=sm+n;double*mass=sm+2*n;
 __shared__ double theta;
 for(int i=threadIdx.x;i<n;i+=blockDim.x){
  double ka=0.,ma=0.,da=0.,qa=0.,qv=0.;
  for(int j=0;j<n;j++){int ij=i*n+j;ka+=K[ij]*q[o+j];ma+=M[ij]*acc[o+j];
   if(o&&!prismatic){da+=D[ij]*q[1+j];qa+=Q[ij]*q[1+j];qv+=Q[ij]*v[1+j];}}
  Da[i]=da;mass[i]=ma;c[i]=prismatic?mt[i]:c0[i]+qa;h[i]=factor*c[i];
  double bv=(o&&!prismatic)?2*v[0]*qv+v[0]*v[0]*(d0[i]+da):0.;
  bias[o+i]=bv;nm[o+i]=ka+bv;
  g[o+i]=endpoint ? weight*nm[o+i]-alpha*old[o+i]+ma+(o?c[i]*acc[0]:0.) : nm[o+i]-alpha/weight*old[o+i]+(ma+(o?c[i]*acc[0]:0.))/weight;
  scale[o+i]=sqrt(fmax((endpoint?weight:1.)*K[i*n+i]+factor*M[i*n+i],1e-20));
 }
 __syncthreads();
 if(threadIdx.x==0){
  double moment=prismatic?total_mass:I0,bv=0.,ma=0.;
  for(int j=0;j<n;j++){if(!prismatic){moment-=2*d0[j]*q[o+j]+q[o+j]*Da[j];bv-=2*v[0]*(d0[j]+Da[j])*v[o+j];}ma+=c[j]*acc[o+j];}
  I[0]=moment;
  if(o){bias[0]=prismatic?0.:bv;nm[0]=bias[0]-load*ramp+guide_k*(q[0]-rest-target*time)+guide_c*(v[0]-target);
   g[0]=endpoint ? weight*nm[0]-alpha*old[0]+moment*acc[0]+ma : nm[0]-alpha/weight*old[0]+(moment*acc[0]+ma)/weight;
   scale[0]=sqrt(fmax(factor*moment+guide_h,1e-20));}
 }
 for(int i=threadIdx.x;i<n;i+=blockDim.x){double value=0.;for(int j=0;j<n;j++)value+=inv[i*n+j]*h[j];u[i]=value;}
 __syncthreads();
 if(threadIdx.x==0){double value=0.;for(int i=0;i<n;i++)value+=h[i]*u[i];sigma[0]=factor*I[0]+guide_h-value;}
 for(int i=threadIdx.x;i<n;i+=blockDim.x){double value=0.;for(int j=0;j<n;j++)value+=inv[i*n+j]*g[o+j];y[i]=value;}
 __syncthreads();
 if(threadIdx.x==0){theta=0.;if(o){double value=0.;for(int j=0;j<n;j++)value+=h[j]*y[j];theta=(g[0]-value)/sigma[0];free[0]=-theta;}}
 __syncthreads();
 for(int i=threadIdx.x;i<n;i+=blockDim.x)free[o+i]=-y[i]+(o?u[i]*theta:0.);
}

extern "C" __global__ void surface_map(const double*X,const double*P,const double*q,const double*s,
 const double*axis,const double*center,double*R,double*x,double*tangent,int nodes,int modes,int o,int revolute,int prismatic){
 int node=blockIdx.x*blockDim.x+threadIdx.x;if(node>=nodes)return;
 double angle=revolute?q[0]:0.,sn=sin(angle),cs=cos(angle),r[9];
 for(int i=0;i<3;i++)for(int j=0;j<3;j++)r[3*i+j]=(i==j?cs:0.)+(1-cs)*axis[i]*axis[j];
 r[1]-=sn*axis[2];r[2]+=sn*axis[1];r[3]+=sn*axis[2];r[5]-=sn*axis[0];r[6]-=sn*axis[1];r[7]+=sn*axis[0];
 double y[3],world[3];
 for(int k=0;k<3;k++){int row=3*node+k;double z=X[row]+s[row]-center[k];for(int j=0;j<modes;j++)z+=P[row*modes+j]*q[o+j];y[k]=z;}
 for(int k=0;k<3;k++){double z=center[k]+(prismatic?q[0]*axis[k]:0.);for(int j=0;j<3;j++)z+=r[3*k+j]*y[j];x[3*node+k]=z;world[k]=z-center[k];}
 if(o)for(int k=0;k<3;k++)tangent[3*node+k]=revolute?axis[(k+1)%3]*world[(k+2)%3]-axis[(k+2)%3]*world[(k+1)%3]:axis[k];
 if(!node)for(int k=0;k<9;k++)R[k]=r[k];
}

extern "C" __global__ void surface_direction(const double*P,const double*dd,const double*ds,const double*R,
 const double*tangent,double*out,int nodes,int modes,int o){
 int node=blockIdx.x*blockDim.x+threadIdx.x;if(node>=nodes)return;double y[3];
 for(int k=0;k<3;k++){int row=3*node+k;double z=ds[row];for(int j=0;j<modes;j++)z+=P[row*modes+j]*dd[o+j];y[k]=z;}
 for(int k=0;k<3;k++){double z=o?tangent[3*node+k]*dd[0]:0.;for(int j=0;j<3;j++)z+=R[3*k+j]*y[j];out[3*node+k]=z;}
}
