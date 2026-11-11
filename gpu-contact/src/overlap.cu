// CUDA translation of the frozen finite-overlap P1 geometry. Forward-mode
// derivatives include clipping, normals, area and barycentric interpolation.
// One derivative direction per thread avoids a 19-component scalar register set.
struct AD {
 double v,d;
 __device__ AD(double a=0.,double b=0.):v(a),d(b){}
};
__device__ AD operator+(AD a,AD b){return AD(a.v+b.v,a.d+b.d);}
__device__ AD operator-(AD a,AD b){return AD(a.v-b.v,a.d-b.d);}
__device__ AD operator-(AD a){return AD(-a.v,-a.d);}
__device__ AD operator*(AD a,AD b){return AD(a.v*b.v,a.d*b.v+a.v*b.d);}
__device__ AD operator/(AD a,AD b){return AD(a.v/b.v,(a.d*b.v-a.v*b.d)/(b.v*b.v));}
__device__ double value(double a){return a;}
__device__ double value(AD a){return a.v;}
__device__ double deriv(double a){return 0.;}
__device__ double deriv(AD a){return a.d;}
__device__ AD root(AD a){double r=sqrt(a.v);return AD(r,a.d/(2*r));}
__device__ double root(double a){return sqrt(a);}
template<class T,int N> struct V {
 T x[N];
 __device__ V(){for(int i=0;i<N;i++)x[i]=T(0.);}
 __device__ T& operator[](int i){return x[i];}
 __device__ const T& operator[](int i)const{return x[i];}
};
template<class T,int N> __device__ V<T,N> operator+(V<T,N>a,V<T,N>b){V<T,N>o;for(int i=0;i<N;i++)o[i]=a[i]+b[i];return o;}
template<class T,int N> __device__ V<T,N> operator-(V<T,N>a,V<T,N>b){V<T,N>o;for(int i=0;i<N;i++)o[i]=a[i]-b[i];return o;}
template<class T,int N> __device__ V<T,N> operator*(V<T,N>a,T b){V<T,N>o;for(int i=0;i<N;i++)o[i]=a[i]*b;return o;}
template<class T,int N> __device__ V<T,N> operator/(V<T,N>a,T b){V<T,N>o;for(int i=0;i<N;i++)o[i]=a[i]/b;return o;}
template<class T,int N> __device__ T dot(V<T,N>a,V<T,N>b){T o(0.);for(int i=0;i<N;i++)o=o+a[i]*b[i];return o;}
template<class T> __device__ V<T,3> cross(V<T,3>a,V<T,3>b){V<T,3>o;for(int i=0;i<3;i++)o[i]=a[(i+1)%3]*b[(i+2)%3]-a[(i+2)%3]*b[(i+1)%3];return o;}
template<class T> __device__ T norm(V<T,3>a){return root(dot(a,a));}
template<class T> __device__ T det(V<T,2>a,V<T,2>b){return a[0]*b[1]-a[1]*b[0];}
template<class T> __device__ V<T,3> bary(V<T,2>p,V<T,2>*t){
 V<T,2>a=t[1]-t[0],b=t[2]-t[0],r=p-t[0];T den=det(a,b),u=det(r,b)/den,v=det(a,r)/den;
 V<T,3>o;o[0]=T(1.)-u-v;o[1]=u;o[2]=v;return o;
}
template<class T> struct Poly{V<T,2>p[12];int count=0;};
template<class T> __device__ Poly<T> clip(Poly<T>p,V<T,3>c){
 Poly<T>o;if(p.count<3)return o;
 V<T,2>prev=p.p[p.count-1];T fp=prev[0]*c[0]+prev[1]*c[1]+c[2]+T(2e-13);
 for(int k=0;k<p.count;k++){
  V<T,2>cur=p.p[k];T fc=cur[0]*c[0]+cur[1]*c[1]+c[2]+T(2e-13);
  if((value(fp)>=0)!=(value(fc)>=0)){if(o.count>=12){o.count=-1;return o;}o.p[o.count++]=prev+(cur-prev)*(fp/(fp-fc));}
  if(value(fc)>=0){if(o.count>=12){o.count=-1;return o;}o.p[o.count++]=cur;}
  prev=cur;fp=fc;
 }
 if(o.count<3)o.count=0;return o;
}
template<class T> struct Frame {
 T length,area2;V<T,3>normal,gc;V<T,2>ss[3],mm[3];Poly<T>poly;
 __device__ bool prepare(V<T,3>*s,V<T,3>*m){
  V<T,3>edge=s[1]-s[0],other=s[2]-s[0],cr=cross(edge,other);length=norm(edge);area2=norm(cr);
  if(value(length)<=0 || value(area2)<=1e-24)return false;
  V<T,3>outward=cr/area2;normal=outward*T(-1.);V<T,3>em=cross(m[1]-m[0],m[2]-m[0]);T am=norm(em);
  if(value(am)<=1e-24 || value(dot(em/am,outward))>=0.)return false;
  V<T,3>e1=edge/length,e2=cross(outward,e1);
  for(int j=0;j<3;j++){V<T,3>a=s[j]-s[0],b=m[j]-s[0];ss[j][0]=dot(a,e1)/length;ss[j][1]=dot(a,e2)/length;mm[j][0]=dot(b,e1)/length;mm[j][1]=dot(b,e2)/length;}
  if(fabs(value(det(mm[1]-mm[0],mm[2]-mm[0])))<1e-12)return false;
  V<T,2>probe[3];probe[1][0]=T(1.);probe[2][1]=T(1.);V<T,3>bm[3];
  for(int j=0;j<3;j++)bm[j]=bary(probe[j],mm);
  poly.count=3;for(int j=0;j<3;j++)poly.p[j]=ss[j];
  for(int k=0;k<3;k++){V<T,3>c;c[0]=bm[1][k]-bm[0][k];c[1]=bm[2][k]-bm[0][k];c[2]=bm[0][k];poly=clip(poly,c);if(poly.count<3)return false;}
  V<T,3>pg;
  for(int j=0;j<3;j++){V<T,3>x=s[0]+e1*(probe[j][0]*length)+e2*(probe[j][1]*length),y;for(int k=0;k<3;k++)y=y+m[k]*bm[j][k];pg[j]=dot(x-y,normal)/length;}
  gc[0]=pg[1]-pg[0];gc[1]=pg[2]-pg[0];gc[2]=pg[0];return true;
 }
};
template<class T> struct Result{
 V<T,3>area,gap;double points[9],normal[3],secondary=0.,minimum=1e100,maximum=-1e100;
 T vg[12];double vp[36];int count=0,samples=0;
 __device__ Result(){for(int i=0;i<9;i++)points[i]=0.;for(int i=0;i<3;i++)normal[i]=0.;}
};
template<class T> __device__ Result<T> evaluate(Frame<T>f,V<T,3>*s,V<T,3>*m,double radius,bool vertex,double negative){
 Result<T>o;
 if(vertex)for(int j=0;j<f.poly.count;j++){
  V<T,3>ns=bary(f.poly.p[j],f.ss),nm=bary(f.poly.p[j],f.mm),xs,xm;
  for(int k=0;k<3;k++){xs=xs+s[k]*ns[k];xm=xm+m[k]*nm[k];}
  o.vg[j]=dot(xs-xm,f.normal);for(int d=0;d<3;d++)o.vp[3*j+d]=value(xm[d]);o.count++;
 }
 Poly<T>poly=f.poly;
 for(int sign=-1;sign<=1;sign+=2){V<T,3>c=f.gc*T(sign);double limit=(sign>0&&negative>0)?negative:radius;c[2]=c[2]+T(limit)/f.length;poly=clip(poly,c);if(poly.count<3)return o;}
 o.secondary=value(f.area2)/2.;for(int k=0;k<3;k++)o.normal[k]=value(f.normal[k]);
 for(int j=1;j+1<poly.count;j++){
  V<T,2>t[3]={poly.p[0],poly.p[j],poly.p[j+1]};T a=det(t[1]-t[0],t[2]-t[0]);if(value(a)<0)a=-a;a=a*f.length*f.length/T(2.);
  if(value(a)<=value(f.area2)*1e-14)continue;
  for(int q=0;q<3;q++){
   V<T,2>p=(t[0]+t[1]+t[2])/T(6.)+t[q]/T(2.);V<T,3>ns=bary(p,f.ss),nm=bary(p,f.mm),xs,xm;
   for(int k=0;k<3;k++){xs=xs+s[k]*ns[k];xm=xm+m[k]*nm[k];}
   T g=dot(xs-xm,f.normal),w=a/T(3.);o.minimum=fmin(o.minimum,value(g));o.maximum=fmax(o.maximum,value(g));o.samples++;
   for(int k=0;k<3;k++){T aw=ns[k]*w;o.area[k]=o.area[k]+aw;o.gap[k]=o.gap[k]+aw*g;for(int d=0;d<3;d++)o.points[3*k+d]+=value(aw)*value(xm[d]);}
  }
 }
 return o;
}
extern "C" __global__ void overlap_values(const double*S,const double*M,int n,double radius,double band,int asymmetric,
 double*area,double*gap,double*points,double*info,int*samples,int*counts,double*vg,double*vp){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 V<double,3>s[3],m[3];for(int k=0;k<9;k++){s[k/3][k%3]=S[9*i+k];m[k/3][k%3]=M[9*i+k];}
 Frame<double>f;if(!f.prepare(s,m))return;
 auto guard=evaluate(f,s,m,radius,true,-1.);if(!guard.samples)return;
 bool near=true;double proximity=asymmetric?radius:band;
 for(int a=0;a<3;a++){double sl=s[0][a],sh=sl,ml=m[0][a],mh=ml;for(int k=1;k<3;k++){sl=fmin(sl,s[k][a]);sh=fmax(sh,s[k][a]);ml=fmin(ml,m[k][a]);mh=fmax(mh,m[k][a]);}near=near&&mh>=sl-proximity&&ml<=sh+proximity;}
 Result<double>patch;if(near)patch=evaluate(f,s,m,band,false,asymmetric?radius:-1.);
 for(int kind=0;kind<2;kind++){
  auto c=kind?patch:guard;int row=2*i+kind;samples[row]=c.samples;
  for(int k=0;k<3;k++){area[3*row+k]=c.area[k];gap[3*row+k]=c.gap[k];info[6*row+k]=c.normal[k];}
  info[6*row+3]=c.minimum;info[6*row+4]=c.maximum;info[6*row+5]=c.secondary;
  for(int k=0;k<9;k++)points[9*row+k]=c.points[k];
 }
 counts[i]=guard.count;for(int j=0;j<guard.count;j++){vg[12*i+j]=guard.vg[j];for(int k=0;k<3;k++)vp[36*i+3*j+k]=guard.vp[3*j+k];}
}
extern "C" __global__ void overlap_derivatives(const double*S,const double*M,const int*selected,int n,double radius,double band,int asymmetric,
 const int*patch_samples,double*partial,double*vertex){
 int j=blockIdx.x;int d=threadIdx.x;if(j>=n||d>=18)return;int i=selected[j];if(!patch_samples[2*i])return;
 V<AD,3>s[3],m[3];for(int k=0;k<9;k++){s[k/3][k%3]=AD(S[9*i+k],k==d);m[k/3][k%3]=AD(M[9*i+k],k+9==d);}
 Frame<AD>f;if(!f.prepare(s,m))return;
 auto guard=evaluate(f,s,m,radius,true,-1.);
 for(int k=0;k<guard.count;k++)vertex[(j*12+k)*18+d]=deriv(guard.vg[k]);
 if(patch_samples[2*i+1]){
  auto patch=evaluate(f,s,m,band,false,asymmetric?radius:-1.);
  for(int k=0;k<3;k++){partial[(j*6+k)*18+d]=deriv(patch.area[k]);partial[(j*6+k+3)*18+d]=deriv(patch.gap[k]);}
 }
}
