// Identical residual nonmonotone line search, now controlled by device state.
// m: phase, accepted evaluations, backtracks, trials, done, error, evaluations,
//    predictor fallback used, reference initialized.
// scalar: reference, step, ceiling; history stores the last five accepted merits.
extern "C" __global__ void record_work(const int*enabled,const int*qp,long long*counts,int candidates){
 if(threadIdx.x||!enabled[0])return;counts[0]++;counts[1]+=candidates;counts[2]+=qp[6];counts[3]+=qp[2];
}
extern "C" __global__ void stage_time(const int*enabled,unsigned long long*start,unsigned long long*total,int stage,int end){
 if(threadIdx.x||!enabled[0])return;unsigned long long tick;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(tick));
 if(end)total[stage]+=tick-start[stage];else start[stage]=tick;
}
extern "C" __global__ void skip_invalid(const int*geometry,int*qp){if(!threadIdx.x&&geometry[0])qp[4]=1;}
extern "C" __global__ void control_step(int*m,double*scalar,double*history,const double*metrics,double*report,
 const int*geometry,const int*qp,double*d,double*s,double*p,const double*committed_d,const double*committed_s,
 double*anchor_d,double*anchor_s,double*anchor_direction,double*anchor_p,
 const double*direction,const double*targetp,const int*rotation_ids,int rotations,int nd,int ns,
 double radius,double absolute_tol,double relative_tol,double static_tol,double gap_tol,int maxiter){
 __shared__ int action; // 1 predictor fallback; 2 new anchor; 3 reduced trial; 4 converged
 __shared__ int accepted;__shared__ double accepted_step;
 int tid=threadIdx.x;
 if(!tid){
  action=0;accepted=0;accepted_step=scalar[1];m[6]++;
  if(geometry[0]&16)m[5]=16;
  else if(geometry[0]){
   if(m[0]==0&&!m[7]){m[7]=1;action=1;}
   else if(m[0]==1){scalar[1]*=.5;m[2]++;action=3;if(m[2]>=30)m[5]=40;}
   else m[5]=42;
  }else if(qp[5])m[5]=100+qp[5];
  else if(metrics[6]<=0.)m[5]=43;
  else{
   if(!m[8]){scalar[0]=fmax(metrics[2],1e-12);m[8]=1;}
   double threshold=absolute_tol+relative_tol*scalar[0];
   double merit=fmax(fmax(metrics[0]/threshold,metrics[1]/static_tol),fmax(metrics[3]/gap_tol,metrics[4]/1e-8));
   report[0]=metrics[0];report[1]=threshold;report[2]=metrics[1];report[3]=metrics[3];report[4]=metrics[4];report[5]=merit;report[6]=metrics[5];
   if(!isfinite(merit)){m[5]=44;}
   else if(m[0]==0||merit<=1.||merit<=(1.-1e-4*scalar[1])*scalar[2]){
    accepted=m[0]==1;m[1]++;m[2]=0;
    if(merit<=1.){m[4]=1;action=4;}
    else if(m[1]>=maxiter)m[5]=41;
    else{
     history[(m[1]-1)%5]=merit;double ceiling=merit;
     for(int j=0;j<5;j++)ceiling=fmax(ceiling,history[j]);scalar[2]=ceiling;
     double step=fmin(1.,.4*radius/fmax(metrics[5],1e-20));
     for(int j=0;j<rotations;j++)step=fmin(step,.025/fmax(fabs(direction[rotation_ids[j]]),1e-20));
     scalar[1]=step;action=2;m[0]=1;
    }
   }else{scalar[1]*=.5;m[2]++;action=3;if(m[2]>=30)m[5]=40;}
  }
  if(m[5])action=0;
  if(action==2||action==3)m[3]++;
 }
 __syncthreads();
 if(accepted)for(int i=tid;i<ns;i+=blockDim.x)p[i]+=accepted_step*(anchor_p[i]-p[i]);
 if(action==1){for(int i=tid;i<nd;i+=blockDim.x)d[i]=committed_d[i];for(int i=tid;i<ns;i+=blockDim.x)s[i]=committed_s[i];}
 if(action==2){
  for(int i=tid;i<nd;i+=blockDim.x){anchor_d[i]=d[i];anchor_direction[i]=direction[i];}
  for(int i=tid;i<ns;i+=blockDim.x){anchor_s[i]=s[i];anchor_p[i]=targetp[i];anchor_direction[nd+i]=direction[nd+i];}
 }
 __syncthreads();
 if(action==2||action==3){
  for(int i=tid;i<nd;i+=blockDim.x)d[i]=anchor_d[i]+scalar[1]*anchor_direction[i];
  for(int i=tid;i<ns;i+=blockDim.x)s[i]=anchor_s[i]+scalar[1]*anchor_direction[nd+i];
 }
}
extern "C" __global__ void commit_step(const int*m,const double*d,const double*s,const double*p,const double*v,
 const double*a,const double*nm,const double*x,double*cd,double*cs,double*cp,double*cv,double*ca,double*cnm,double*cx,int nd,int ns){
 if(!m[4]||m[5])return;int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<nd){cd[i]=d[i];cv[i]=v[i];ca[i]=a[i];cnm[i]=nm[i];}
 if(i<ns){cs[i]=s[i];cp[i]=p[i];cx[i]=x[i];}
}
