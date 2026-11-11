// FP64 Surface-CMS/HHT equations. One warp per dense row / surface node.
__device__ double warp_sum(double v){
 for(int d=16;d;d/=2)v+=__shfl_down_sync(0xffffffff,v,d);return v;
}
#define MODAL_ARGS const double*K,const double*M,const double*Q,const double*D, \
 const double*inv,const double*c0,const double*d0,const double*mt, \
 const double*q,const double*v,const double*acc,const double*old, \
 double*c,double*h,double*u,double*I,double*sigma,double*bias, \
 double*nm,double*g,double*scale,double*free,int n,int o,int prismatic,int endpoint, \
 double I0,double total_mass,double factor,double weight,double alpha, \
 double guide_k,double guide_c,double guide_h,double target,double rest,double load, \
 const double*clock,double ramp_duration,double*Da,double*y

extern "C" __global__ void modal_rows(MODAL_ARGS){
 int row=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;if(row>=n)return;
 double ka=0.,ma=0.,da=0.,qa=0.,qv=0.;
 for(int j=lane;j<n;j+=32){int ij=row*n+j;ka+=K[ij]*q[o+j];ma+=M[ij]*acc[o+j];
  if(o&&!prismatic){da+=D[ij]*q[1+j];qa+=Q[ij]*q[1+j];qv+=Q[ij]*v[1+j];}}
 ka=warp_sum(ka);ma=warp_sum(ma);da=warp_sum(da);qa=warp_sum(qa);qv=warp_sum(qv);
 if(lane)return;
 Da[row]=da;c[row]=prismatic?mt[row]:c0[row]+qa;h[row]=factor*c[row];
 double bv=(o&&!prismatic)?2*v[0]*qv+v[0]*v[0]*(d0[row]+da):0.;
 bias[o+row]=bv;nm[o+row]=ka+bv;
 g[o+row]=endpoint ? weight*nm[o+row]-alpha*old[o+row]+ma+(o?c[row]*acc[0]:0.) : nm[o+row]-alpha/weight*old[o+row]+(ma+(o?c[row]*acc[0]:0.))/weight;
 scale[o+row]=sqrt(fmax((endpoint?weight:1.)*K[row*n+row]+factor*M[row*n+row],1e-20));
}
extern "C" __global__ void modal_reference(MODAL_ARGS){
 if(threadIdx.x)return;
 double moment=prismatic?total_mass:I0,bv=0.,ma=0.;
 for(int j=0;j<n;j++){if(!prismatic){moment-=2*d0[j]*q[o+j]+q[o+j]*Da[j];bv-=2*v[0]*(d0[j]+Da[j])*v[o+j];}ma+=c[j]*acc[o+j];}
 I[0]=moment;
 if(o){double time=clock[0],ramp=fmin(1.,time/ramp_duration);
  bias[0]=prismatic?0.:bv;nm[0]=bias[0]-load*ramp+guide_k*(q[0]-rest-target*time)+guide_c*(v[0]-target);
  g[0]=endpoint ? weight*nm[0]-alpha*old[0]+moment*acc[0]+ma : nm[0]-alpha/weight*old[0]+(moment*acc[0]+ma)/weight;
  scale[0]=sqrt(fmax(factor*moment+guide_h,1e-20));}
}
extern "C" __global__ void modal_inverse(MODAL_ARGS){
 int row=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;if(row>=n)return;
 double uh=0.,yg=0.;for(int j=lane;j<n;j+=32){double a=inv[row*n+j];uh+=a*h[j];yg+=a*g[o+j];}
 uh=warp_sum(uh);yg=warp_sum(yg);if(!lane){u[row]=uh;y[row]=yg;}
}
extern "C" __global__ void modal_finish(MODAL_ARGS){
 __shared__ double theta;
 if(threadIdx.x==0){double hu=0.,hy=0.;for(int j=0;j<n;j++){hu+=h[j]*u[j];hy+=h[j]*y[j];}
  sigma[0]=factor*I[0]+guide_h-hu;theta=o?(g[0]-hy)/sigma[0]:0.;if(o)free[0]=-theta;}
 __syncthreads();for(int j=threadIdx.x;j<n;j+=blockDim.x)free[o+j]=-y[j]+(o?u[j]*theta:0.);
}
extern "C" __global__ void parallel_map(const double*X,const double*P,const double*q,const double*s,
 const double*axis,const double*center,double*R,double*x,double*tangent,int nodes,int modes,int o,int revolute,int prismatic){
 int node=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;if(node>=nodes)return;
 double y[3];for(int k=0;k<3;k++){double z=0.;for(int j=lane;j<modes;j+=32)z+=P[(3*node+k)*modes+j]*q[o+j];y[k]=warp_sum(z);}
 if(lane)return;
 double angle=revolute?q[0]:0.,sn=sin(angle),cs=cos(angle),r[9];
 for(int i=0;i<3;i++)for(int j=0;j<3;j++)r[3*i+j]=(i==j?cs:0.)+(1-cs)*axis[i]*axis[j];
 r[1]-=sn*axis[2];r[2]+=sn*axis[1];r[3]+=sn*axis[2];r[5]-=sn*axis[0];r[6]-=sn*axis[1];r[7]+=sn*axis[0];
 for(int k=0;k<3;k++)y[k]+=X[3*node+k]+s[3*node+k]-center[k];
 double world[3];for(int k=0;k<3;k++){double z=center[k]+(prismatic?q[0]*axis[k]:0.);for(int j=0;j<3;j++)z+=r[3*k+j]*y[j];x[3*node+k]=z;world[k]=z-center[k];}
 if(o)for(int k=0;k<3;k++)tangent[3*node+k]=revolute?axis[(k+1)%3]*world[(k+2)%3]-axis[(k+2)%3]*world[(k+1)%3]:axis[k];
 if(!node)for(int k=0;k<9;k++)R[k]=r[k];
}
extern "C" __global__ void parallel_direction(const double*P,const double*dd,const double*ds,const double*R,
 const double*tangent,double*out,int nodes,int modes,int o){
 int node=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;if(node>=nodes)return;
 double y[3];for(int k=0;k<3;k++){double z=0.;for(int j=lane;j<modes;j+=32)z+=P[(3*node+k)*modes+j]*dd[o+j];y[k]=warp_sum(z);}
 if(lane)return;for(int k=0;k<3;k++)y[k]+=ds[3*node+k];
 for(int k=0;k<3;k++){double z=o?tangent[3*node+k]*dd[0]:0.;for(int j=0;j<3;j++)z+=R[3*k+j]*y[j];out[3*node+k]=z;}
}
