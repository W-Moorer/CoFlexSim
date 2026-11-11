"""Exact degree-two polygon moments for the existing linear contact integrals.

Clipping, finite support, area rejection and guard vertices are unchanged.
This replaces three-point quadrature arithmetic by its closed-form moments;
it introduces neither sampling nor a new contact pressure discretization.
"""

SPECIALIZATION = r'''
__device__ double moment_product(const double*a,const double*b,const double*M){
 return a[0]*b[0]*M[0]+(a[0]*b[1]+a[1]*b[0])*M[1]
  +(a[0]*b[2]+a[2]*b[0])*M[2]+a[1]*b[1]*M[3]
  +(a[1]*b[2]+a[2]*b[1])*M[4]+a[2]*b[2]*M[5];
}
template<> __device__ Result<double> evaluate<double>(Frame<double>f,V<double,3>*s,V<double,3>*m,
 double radius,bool vertex,double negative){
 Result<double>o;
 if(vertex)for(int j=0;j<f.poly.count;j++){
  V<double,3>ns=bary(f.poly.p[j],f.ss),nm=bary(f.poly.p[j],f.mm),xs,xm;
  for(int k=0;k<3;k++){xs=xs+s[k]*ns[k];xm=xm+m[k]*nm[k];}
  o.vg[j]=dot(xs-xm,f.normal);for(int d=0;d<3;d++)o.vp[3*j+d]=xm[d];o.count++;
 }
 Poly<double>poly=f.poly;
 for(int sign=-1;sign<=1;sign+=2){V<double,3>c=f.gc*double(sign);
  double limit=(sign>0&&negative>0)?negative:radius;
  c[2]+=limit/f.length;poly=clip(poly,c);if(poly.count<3)return o;}
 o.secondary=f.area2/2.;for(int k=0;k<3;k++)o.normal[k]=f.normal[k];
 double M[6]={0.,0.,0.,0.,0.,0.};
 for(int j=1;j+1<poly.count;j++){
  V<double,2>t[3]={poly.p[0],poly.p[j],poly.p[j+1]};
  double a=fabs(det(t[1]-t[0],t[2]-t[0]))*f.length*f.length/2.;
  if(a<=f.area2*1e-14)continue;
  double sx=0.,sy=0.,xx=0.,xy=0.,yy=0.;
  for(int k=0;k<3;k++){sx+=t[k][0];sy+=t[k][1];xx+=t[k][0]*t[k][0];xy+=t[k][0]*t[k][1];yy+=t[k][1]*t[k][1];}
  M[0]+=a;M[1]+=a*sx/3.;M[2]+=a*sy/3.;
  M[3]+=a*(sx*sx+xx)/12.;M[4]+=a*(sx*sy+xy)/12.;M[5]+=a*(sy*sy+yy)/12.;
  for(int q=0;q<3;q++){
   V<double,2>p=(t[0]+t[1]+t[2])/6.+t[q]/2.;
   double g=f.length*(f.gc[0]*p[0]+f.gc[1]*p[1]+f.gc[2]);
   o.minimum=fmin(o.minimum,g);o.maximum=fmax(o.maximum,g);o.samples++;
  }
 }
 if(!o.samples)return o;
 V<double,2>probe[3];probe[1][0]=1.;probe[2][1]=1.;
 V<double,3>ns[3],nm[3];
 for(int j=0;j<3;j++){ns[j]=bary(probe[j],f.ss);nm[j]=bary(probe[j],f.mm);}
 double X[3][3],g[3]={f.length*f.gc[2],f.length*f.gc[0],f.length*f.gc[1]};
 for(int d=0;d<3;d++){
  double v[3]={0.,0.,0.};
  for(int j=0;j<3;j++)for(int k=0;k<3;k++)v[j]+=m[k][d]*nm[j][k];
  X[d][0]=v[0];X[d][1]=v[1]-v[0];X[d][2]=v[2]-v[0];
 }
 for(int k=0;k<3;k++){
  double N[3]={ns[0][k],ns[1][k]-ns[0][k],ns[2][k]-ns[0][k]};
  o.area[k]=N[0]*M[0]+N[1]*M[1]+N[2]*M[2];
  o.gap[k]=moment_product(N,g,M);
  for(int d=0;d<3;d++)o.points[3*k+d]=moment_product(N,X[d],M);
 }
 return o;
}
'''


def specialize(source):
    marker='extern "C" __global__ void overlap_values('
    assert source.count(marker)==1
    return source.replace(marker,SPECIALIZATION+'\n'+marker)
