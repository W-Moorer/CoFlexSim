// Differentiated finite-overlap P1 weak gaps. Derivatives include clipping,
// projection, normals, shape functions, area and support normalization (Python).
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <unsupported/Eigen/AutoDiff>
#include <array>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>
#include "contact_bvh.h"
namespace py=pybind11;
using D=Eigen::AutoDiffScalar<Eigen::Matrix<double,18,1>>;
template<class S> using V2=Eigen::Matrix<S,2,1>;
template<class S> using V3=Eigen::Matrix<S,3,1>;
template<class S> double val(const S& a){return a;}
template<class Der> double val(const Eigen::AutoDiffScalar<Der>& a){return a.value();}
template<class S> S positive_abs(const S&a){return val(a)<0?-a:a;}
template<class S> S norm3(const V3<S>&a){using std::sqrt;return sqrt(a.dot(a));}
template<class S> S det2(const V2<S>&a,const V2<S>&b){return a[0]*b[1]-a[1]*b[0];}
template<class S> V3<S> bary(const V2<S>&p,const std::array<V2<S>,3>&t){
 V2<S>a=t[1]-t[0],b=t[2]-t[0],r=p-t[0];S det=det2(a,b),u=det2(r,b)/det,v=det2(a,r)/det;
 V3<S>q;q<<S(1)-u-v,u,v;return q;
}
template<class S> std::vector<V2<S>> clip(const std::vector<V2<S>>&p,const V3<S>&c,std::vector<int>*edges=nullptr,int boundary=0){
 std::vector<int> labels;
 std::vector<V2<S>>out;if(p.size()<3)return out;
 // Classify and intersect the SAME roundoff-expanded halfspace. Using
 // f>=-eps for classification but f=0 for intersection can extrapolate
 // outside the segment when both endpoint values are negative near -eps.
 auto eval=[&](const V2<S>&v)->S{return v[0]*c[0]+v[1]*c[1]+c[2]+S(2e-13);};
 V2<S>prev=p.back();S fp=eval(prev);
 size_t k=0;int last=edges?edges->back():0;
 for(const auto&cur:p){int label=edges?(*edges)[k++]:0;S fc=eval(cur);if((val(fp)>=0)!=(val(fc)>=0)){out.push_back(prev+(cur-prev)*(fp/(fp-fc)));if(edges)labels.push_back((last&label)|boundary);}
  if(val(fc)>=0){out.push_back(cur);if(edges)labels.push_back(label);}last=label;prev=cur;fp=fc;}
 if(edges)edges->swap(labels);
 if(out.size()<3)out.clear();return out;
}
template<class S> struct Contribution{
 V3<S>area=V3<S>::Zero(),gap=V3<S>::Zero();
 Eigen::Matrix<double,3,3>points=Eigen::Matrix<double,3,3>::Zero();
 Eigen::Vector3d normal=Eigen::Vector3d::Zero();
 double min_gap=1e100,max_gap=-1e100,secondary_area=0;int samples=0;
 S vertex_gap=S(1e100);Eigen::Vector3d vertex_point=Eigen::Vector3d::Zero();
 std::vector<S>vertex_gaps;std::vector<Eigen::Vector3d>vertex_points;
};
using Clock=std::chrono::steady_clock;
// Fine-grained clock reads are compiled only in the standalone profiling binary.
#ifdef COFLEX_PROFILE_CLIP
thread_local double profiled_physical_clip_s=0.;
struct ClipScope {
 Clock::time_point start=Clock::now();
 ~ClipScope(){profiled_physical_clip_s+=std::chrono::duration<double>(Clock::now()-start).count();}
};
#endif
template<class S> struct Overlap {
 S length,area2; V3<S>n,gc; std::array<V2<S>,3>ss,mm; std::vector<V2<S>>poly;
 bool prepare(const std::array<V3<S>,3>&s,const std::array<V3<S>,3>&m,std::vector<int>*labels=nullptr){
 V3<S>edge=s[1]-s[0],other=s[2]-s[0],cross=edge.cross(other);length=norm3(edge);area2=norm3(cross);
 if(val(length)<=0||val(area2)<=1e-24)return false;
 V3<S>outward=cross/area2; n=-outward; V3<S>em=(m[1]-m[0]).cross(m[2]-m[0]);S am=norm3(em);
 // Reject only co-facing surfaces. An arbitrary obliqueness cutoff (formerly
 // -0.25) discards finite projected area and makes the normalized weak gap
 // discontinuous under infinitesimal rotation. At the hemisphere boundary
 // the main triangle's projected area tends to zero continuously; the
 // determinant guard below handles the exactly edge-on configuration.
 if(val(am)<=1e-24||val((em/am).dot(outward))>=0.)return false;
 V3<S>e1=edge/length,e2=outward.cross(e1);
 for(int j=0;j<3;j++){V3<S>a=s[j]-s[0],b=m[j]-s[0];ss[j]<<a.dot(e1)/length,a.dot(e2)/length;mm[j]<<b.dot(e1)/length,b.dot(e2)/length;}
 V2<S>m1=mm[1]-mm[0],m2=mm[2]-mm[0];if(std::abs(val(det2(m1,m2)))<1e-12)return false;
 std::array<V2<S>,3>probe;probe[0]<<S(0),S(0);probe[1]<<S(1),S(0);probe[2]<<S(0),S(1);
 std::array<V3<S>,3>bm;for(int j=0;j<3;j++)bm[j]=bary(probe[j],mm);
 poly.assign(ss.begin(),ss.end());if(labels)*labels={6,5,3};
 {
#ifdef COFLEX_PROFILE_CLIP
 ClipScope profile_scope;
#endif
 for(int k=0;k<3;k++){V3<S>c;c<<bm[1][k]-bm[0][k],bm[2][k]-bm[0][k],bm[0][k];poly=clip(poly,c,labels,1<<(3+k));if(poly.empty())return false;}
 }
 V3<S>pg;
 for(int j=0;j<3;j++){V3<S>x=s[0]+probe[j][0]*length*e1+probe[j][1]*length*e2,y=V3<S>::Zero();for(int k=0;k<3;k++)y+=bm[j][k]*m[k];pg[j]=(x-y).dot(n)/length;}
 gc<<pg[1]-pg[0],pg[2]-pg[0],pg[0];
 return true;
 }
 Contribution<S> evaluate(const std::array<V3<S>,3>&s,const std::array<V3<S>,3>&m,double radius,bool vertex_rule,double negative_radius=-1.)const{
 Contribution<S>out;
 // The gap is affine over the finite overlap polygon. Its minimum is at a
 // physical polygon vertex. Compute BEFORE search-radius clipping so an
 // artificial search-cut point cannot become a load-bearing contact feature.
 if(vertex_rule)for(const auto&p:this->poly){
  V3<S>ns=bary(p,ss),nm=bary(p,mm),xs=V3<S>::Zero(),xm=V3<S>::Zero();
  for(int k=0;k<3;k++){xs+=ns[k]*s[k];xm+=nm[k]*m[k];}
  S g=(xs-xm).dot(n);Eigen::Vector3d location;for(int d=0;d<3;d++)location[d]=val(xm[d]);
  out.vertex_gaps.push_back(g);out.vertex_points.push_back(location);
  if(val(g)<val(out.vertex_gap)){
   out.vertex_gap=g;for(int d=0;d<3;d++)out.vertex_point[d]=val(xm[d]);
  }
 }
 auto poly=this->poly;
 for(double sign:{-1.,1.}){V3<S>c=gc*S(sign);double limit=(sign>0. && negative_radius>0.)?negative_radius:radius;c[2]+=S(limit)/length;poly=clip(poly,c);if(poly.empty())return out;}
 out.secondary_area=val(area2)/2;for(int k=0;k<3;k++)out.normal[k]=val(n[k]);
 for(size_t j=1;j+1<poly.size();j++){
  std::array<V2<S>,3>t={poly[0],poly[j],poly[j+1]};V2<S>a=t[1]-t[0],b=t[2]-t[0];S area=positive_abs(det2(a,b))*length*length/S(2);
  if(val(area)<=val(area2)*1e-14)continue;
  for(int q=0;q<3;q++){V2<S>p=(t[0]+t[1]+t[2])/S(6)+t[q]/S(2);V3<S>ns=bary(p,ss),nm=bary(p,mm),xs=V3<S>::Zero(),xm=V3<S>::Zero();
   for(int k=0;k<3;k++){xs+=ns[k]*s[k];xm+=nm[k]*m[k];}
   S g=(xs-xm).dot(n),w=area/S(3);out.min_gap=std::min(out.min_gap,val(g));out.max_gap=std::max(out.max_gap,val(g));out.samples++;
   for(int k=0;k<3;k++){S aw=ns[k]*w;out.area[k]+=aw;out.gap[k]+=aw*g;for(int d=0;d<3;d++)out.points(k,d)+=val(aw)*val(xm[d]);}
  }
 }
 return out;
}
};
template<class S> Contribution<S> integrate(const std::array<V3<S>,3>&s,const std::array<V3<S>,3>&m,double radius,bool vertex_rule=false,double*profile=nullptr,double negative_radius=-1.){
 Overlap<S> overlap; auto tick=profile?Clock::now():Clock::time_point{};
 bool valid=overlap.prepare(s,m);
 if(profile)*profile+=std::chrono::duration<double>(Clock::now()-tick).count();
 return valid?overlap.evaluate(s,m,radius,vertex_rule,negative_radius):Contribution<S>{};
}
py::dict batch(const Eigen::MatrixXd& secondary,const Eigen::MatrixXd& main,double radius,bool derivatives,int threads,bool vertex_rule,bool profile=false){
 using Clock=std::chrono::steady_clock;auto start=Clock::now();
 if(secondary.rows()!=main.rows()||secondary.cols()!=9||main.cols()!=9||radius<=0)throw std::invalid_argument("Expected matching N x 9 triangles and positive search radius");
 if(threads<1)throw std::invalid_argument("Positive contact thread count required");
 int count=int(secondary.rows());Eigen::MatrixXd area=Eigen::MatrixXd::Zero(count,3),gap=area,da=Eigen::MatrixXd::Zero(derivatives?count:0,54),dg=da;
 Eigen::MatrixXd info=Eigen::MatrixXd::Zero(count,6),points=Eigen::MatrixXd::Zero(count,9);Eigen::VectorXi samples=Eigen::VectorXi::Zero(count);
 Eigen::VectorXd vertex_gap=Eigen::VectorXd::Constant(vertex_rule?count:0,1e100);
 Eigen::MatrixXd vertex_derivative=Eigen::MatrixXd::Zero(vertex_rule&&derivatives?count:0,18),vertex_point=Eigen::MatrixXd::Zero(vertex_rule?count:0,3);
 Eigen::MatrixXd vertex_gaps=Eigen::MatrixXd::Constant(vertex_rule?count:0,12,1e100),vertex_points=Eigen::MatrixXd::Zero(vertex_rule?count:0,36),vertex_derivatives=Eigen::MatrixXd::Zero(vertex_rule&&derivatives?count:0,216);
 Eigen::VectorXi vertex_counts=Eigen::VectorXi::Zero(vertex_rule?count:0);
 if(profile && threads!=1)throw std::invalid_argument("Profile overlap on one thread");
 double scalar_clip=0,scalar_overlap=0,ad_overlap=0;auto allocated=Clock::now();double scalar_seconds=0.,derivative_seconds=0.;
 {
 py::gil_scoped_release release;
#ifdef COFLEX_PROFILE_CLIP
 profiled_physical_clip_s=0.;
#endif
 auto scalar_start=Clock::now();
 #pragma omp parallel for num_threads(threads) schedule(static) if(count>=32 && threads>1)
 for(int i=0;i<count;i++){
   std::array<V3<double>,3>s,m;
   for(int k=0;k<9;k++){s[k/3][k%3]=secondary(i,k);m[k/3][k%3]=main(i,k);}
   auto c=integrate(s,m,radius,vertex_rule,profile?&scalar_overlap:nullptr);samples[i]=c.samples;
   if(vertex_rule){vertex_gap[i]=c.vertex_gap;vertex_point.row(i)=c.vertex_point;vertex_counts[i]=int(c.vertex_gaps.size());
    for(int j=0;j<int(c.vertex_gaps.size())&&j<12;j++){vertex_gaps(i,j)=c.vertex_gaps[j];vertex_points.block<1,3>(i,3*j)=c.vertex_points[j];}}
   // Empty rows preserve the historical derivative path's all-zero payload.
   if(!derivatives||c.samples)for(int k=0;k<3;k++){area(i,k)=c.area[k];gap(i,k)=c.gap[k];info(i,k)=c.normal[k];for(int d=0;d<3;d++)points(i,3*k+d)=c.points(k,d);}
   info(i,3)=c.min_gap;info(i,4)=c.max_gap;if(!derivatives||c.samples)info(i,5)=c.secondary_area;
 }
 scalar_seconds=std::chrono::duration<double>(Clock::now()-scalar_start).count();
#ifdef COFLEX_PROFILE_CLIP
 scalar_clip=profiled_physical_clip_s;profiled_physical_clip_s=0.;
#endif
 auto derivative_start=Clock::now();
 if(derivatives){
  std::vector<int> active;active.reserve(count);for(int i=0;i<count;i++)if(samples[i])active.push_back(i);
  #pragma omp parallel for num_threads(threads) schedule(static) if(active.size()>=32 && threads>1)
  for(int row=0;row<int(active.size());row++){
   int i=active[row];
   std::array<V3<D>,3>s,m;
   for(int k=0;k<9;k++){s[k/3][k%3]=D(secondary(i,k));s[k/3][k%3].derivatives()[k]=1;m[k/3][k%3]=D(main(i,k));m[k/3][k%3].derivatives()[9+k]=1;}
   auto c=integrate(s,m,radius,vertex_rule,profile?&ad_overlap:nullptr);samples[i]=c.samples;
   if(vertex_rule){vertex_gap[i]=c.vertex_gap.value();vertex_derivative.row(i)=c.vertex_gap.derivatives().transpose();vertex_point.row(i)=c.vertex_point;
    for(int j=0;j<int(c.vertex_gaps.size())&&j<12;j++){vertex_gaps(i,j)=c.vertex_gaps[j].value();vertex_derivatives.block<1,18>(i,18*j)=c.vertex_gaps[j].derivatives().transpose();}}
   for(int k=0;k<3;k++){area(i,k)=c.area[k].value();gap(i,k)=c.gap[k].value();for(int d=0;d<18;d++){da(i,18*k+d)=c.area[k].derivatives()[d];dg(i,18*k+d)=c.gap[k].derivatives()[d];}
    info(i,k)=c.normal[k];for(int d=0;d<3;d++)points(i,3*k+d)=c.points(k,d);}
   info(i,3)=c.min_gap;info(i,4)=c.max_gap;info(i,5)=c.secondary_area;
  }
 }
 derivative_seconds=std::chrono::duration<double>(Clock::now()-derivative_start).count();
 }
 py::dict out;out["area"]=area;out["weighted_gap"]=gap;out["area_derivative"]=da;out["gap_derivative"]=dg;out["info"]=info;out["points"]=points;out["samples"]=samples;
 if(vertex_rule){out["vertex_gap"]=vertex_gap;out["vertex_derivative"]=vertex_derivative;out["vertex_point"]=vertex_point;
  out["vertex_counts"]=vertex_counts;out["vertex_gaps"]=vertex_gaps;out["vertex_points"]=vertex_points;out["vertex_derivatives"]=vertex_derivatives;}
 py::dict timing;timing["native_allocation"]=std::chrono::duration<double>(allocated-start).count();timing["scalar_geometry"]=scalar_seconds;timing["geometry_derivatives"]=derivative_seconds;out["timings_seconds"]=timing;
#ifdef COFLEX_PROFILE_CLIP
 out["scalar_clip_s"]=scalar_clip;out["ad_clip_s"]=profiled_physical_clip_s;
#endif
 out["scalar_overlap_s"]=scalar_overlap;out["ad_overlap_s"]=ad_overlap;return out;
}
#include "shared_overlap.h"
#ifdef COFLEX_PROFILE_CLIP
PYBIND11_MODULE(coflex_mortar_profile,m){
#else
PYBIND11_MODULE(coflex_mortar_native,m){
#endif
 bind_shared_overlap(m);
 m.def("batch",&batch,py::arg("secondary"),py::arg("main"),py::arg("radius"),py::arg("derivatives")=true,py::arg("threads")=1,py::arg("vertex_rule")=false,py::arg("profile")=false);
 m.attr("geometry_id")="finite_overlap_P1_normalized_gap_consistent_clip_v3";
 m.attr("execution_id")="parallel_scalar_then_derivative_v1";
 py::class_<ContactAABBTree>(m,"AABBTree").def(py::init<const Eigen::MatrixXd&,const Eigen::MatrixXd&,int>(),py::arg("low"),py::arg("high"),py::arg("leaf_size")=8)
  .def("refit",&ContactAABBTree::refit).def("query",&ContactAABBTree::query).def_property_readonly("bytes",&ContactAABBTree::bytes);
}
