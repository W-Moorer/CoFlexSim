extern "C" __global__ void triangle_boxes(const double*x,const int*f,double*lo,double*hi,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 for(int a=0;a<3;a++){double l=x[3*f[3*i]+a],h=l;for(int k=1;k<3;k++){double v=x[3*f[3*i+k]+a];l=fmin(l,v);h=fmax(h,v);}lo[3*i+a]=l;hi[3*i+a]=h;}
}
extern "C" __global__ void leaf_boxes(const int*nodes,const int*order,const int*ids,int n,
 const double*tl,const double*th,double*lo,double*hi){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=n)return;int i=ids[k],start=nodes[4*i+2],count=nodes[4*i+3];
 for(int a=0;a<3;a++){double l=1e300,h=-1e300;for(int j=0;j<count;j++){int t=order[start+j];l=fmin(l,tl[3*t+a]);h=fmax(h,th[3*t+a]);}lo[3*i+a]=l;hi[3*i+a]=h;}
}
extern "C" __global__ void inner_boxes(const int*nodes,const int*ids,int n,double*lo,double*hi){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=n)return;int i=ids[k],l=nodes[4*i],r=nodes[4*i+1];
 for(int a=0;a<3;a++){lo[3*i+a]=fmin(lo[3*l+a],lo[3*r+a]);hi[3*i+a]=fmax(hi[3*l+a],hi[3*r+a]);}
}
extern "C" __global__ void query_boxes(const int*nodes,const int*order,const double*lo,const double*hi,
 const double*tl,const double*th,const double*ql,const double*qh,int n,double radius,int cap,int*out,int*counts,int*overflow){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int stack[64],top=0,count=0;stack[top++]=0;
 while(top){int j=stack[--top];bool hit=true;for(int a=0;a<3;a++)hit=hit&&hi[3*j+a]>=ql[3*i+a]-radius&&lo[3*j+a]<=qh[3*i+a]+radius;
  if(!hit)continue;
  if(nodes[4*j+3]){
   for(int k=0;k<nodes[4*j+3];k++){int t=order[nodes[4*j+2]+k];bool fine=true;
    for(int a=0;a<3;a++)fine=fine&&th[3*t+a]>=ql[3*i+a]-radius&&tl[3*t+a]<=qh[3*i+a]+radius;
    if(fine){if(count<cap)out[i*cap+count]=t;else atomicExch(overflow,1);count++;}
   }
  }else{if(top+2>64){atomicExch(overflow,2);return;}stack[top++]=nodes[4*j];stack[top++]=nodes[4*j+1];}
 }
 counts[i]=count;
}
