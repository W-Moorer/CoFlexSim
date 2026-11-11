// Two radii share the physical overlap. A guard vertex retains its intersecting
// edges so a displacement can be tested without creating its 18-entry Jacobian.
using D1=Eigen::AutoDiffScalar<Eigen::Matrix<double,1,1>>;
using Tri=std::array<Eigen::Vector3d,3>;
struct VertexKinematics {
 Eigen::Vector3d slave,main,h,normal_weights;
 double gap;
};
struct SharedFace {
 int input=0;
 Tri s,m;
 Contribution<double> guard,patch;
 std::vector<VertexKinematics> vertices;
 bool near=false;
};

struct SharedFaceBatch {
 std::vector<SharedFace> faces;
 double wide,band;
 bool retain_negative;
 int common_prepares=0,near_tests=0;
 SharedFaceBatch(const Eigen::MatrixXd&S,const Eigen::MatrixXd&M,double radius,double narrow,bool linearize,bool asymmetric=false)
 :wide(radius),band(narrow),retain_negative(asymmetric){
  if(S.rows()!=M.rows()||S.cols()!=9||M.cols()!=9||!(wide>band&&band>0))throw std::invalid_argument("Invalid bounded face batch");
  for(int i=0;i<S.rows();i++){
   SharedFace f; f.input=i;
   for(int k=0;k<9;k++){f.s[k/3][k%3]=S(i,k);f.m[k/3][k%3]=M(i,k);}
   Overlap<double> frame; std::vector<int> edges;common_prepares++;
   if(!frame.prepare(f.s,f.m,linearize?&edges:nullptr))continue;
   f.guard=frame.evaluate(f.s,f.m,wide,true);
   if(!f.guard.samples)continue;
   bool near=true;
   for(int axis=0;axis<3;axis++){
    double s0=f.s[0][axis],s1=s0,m0=f.m[0][axis],m1=m0;
    for(int k=1;k<3;k++){s0=std::min(s0,f.s[k][axis]);s1=std::max(s1,f.s[k][axis]);m0=std::min(m0,f.m[k][axis]);m1=std::max(m1,f.m[k][axis]);}
    double proximity=retain_negative?wide:band;
    near=near && m1>=s0-proximity && m0<=s1+proximity;
   }
   if(near){near_tests++;f.patch=frame.evaluate(f.s,f.m,band,false,retain_negative?wide:-1.);f.near=f.patch.samples>0;}
   if(linearize){
    Eigen::Vector3d normal=frame.n;
    std::array<Eigen::Vector3d,3>grad;
    for(int k=0;k<3;k++)grad[k]=(-normal).cross(f.s[(k+2)%3]-f.s[(k+1)%3])/frame.area2;
    for(size_t j=0;j<edges.size();j++){
     int sm=edges[j]&7,mm=(edges[j]>>3)&7;
     auto edge=[](const Tri&t,int mask)->Eigen::Vector3d{
      int k=mask==1?0:(mask==2?1:2);return t[(k+2)%3]-t[(k+1)%3];};
     Eigen::Vector3d h;
     if(!sm)h=normal;
     else if(!mm)h=(f.m[1]-f.m[0]).cross(f.m[2]-f.m[0]);
     else h=edge(f.s,sm).cross(edge(f.m,mm));
     double den=h.dot(normal);
     if(std::abs(den)<1.e-14*h.norm()||den==0)
      throw std::runtime_error("Singular finite-overlap feature sensitivity");
     h/=den;
     VertexKinematics v;v.slave=bary(frame.poly[j],frame.ss);v.main=bary(frame.poly[j],frame.mm);v.h=h;
     v.gap=f.guard.vertex_gaps[j];
     for(int k=0;k<3;k++)v.normal_weights[k]=v.gap*h.dot(grad[k]);
     f.vertices.push_back(v);
    }
   }
   faces.push_back(std::move(f));
  }
 }
 py::dict values(bool pressure)const{
  int n=(int)faces.size();Eigen::MatrixXd area(n,3),gap(n,3),points(n,9),info(n,6);
  Eigen::VectorXi input(n),samples(n),counts(n);Eigen::MatrixXd vg=Eigen::MatrixXd::Zero(n,12),vp=Eigen::MatrixXd::Zero(n,36);
  for(int i=0;i<n;i++){
   const auto&f=faces[i];const auto&c=pressure?f.patch:f.guard;input[i]=f.input;samples[i]=c.samples;counts[i]=(int)c.vertex_gaps.size();
   area.row(i)=c.area.transpose();gap.row(i)=c.gap.transpose();
   info.row(i)<<c.normal.transpose(),c.min_gap,c.max_gap,c.secondary_area;
   for(int j=0;j<3;j++)points.block<1,3>(i,3*j)=c.points.row(j);
   if(counts[i]>12)throw std::runtime_error("Physical overlap vertex capacity exceeded");
   for(int j=0;j<counts[i];j++){vg(i,j)=c.vertex_gaps[j];vp.block<1,3>(i,3*j)=c.vertex_points[j].transpose();}
  }
  py::dict out;out["input"]=input;out["samples"]=samples;out["area"]=area;out["weighted_gap"]=gap;out["points"]=points;out["info"]=info;
  if(!pressure){out["vertex_counts"]=counts;out["vertex_gaps"]=vg;out["vertex_points"]=vp;}return out;
 }
 Eigen::MatrixXd vertex_rows(const Eigen::VectorXi&pairs,const Eigen::VectorXi&slots)const{
  if(pairs.size()!=slots.size())throw std::invalid_argument("Mismatched vertex row IDs");
  Eigen::MatrixXd rows(pairs.size(),18);
  for(int j=0;j<pairs.size();j++){
   const auto&f=faces.at(pairs[j]);const auto&v=f.vertices.at(slots[j]);
   for(int k=0;k<3;k++){
    rows.block<1,3>(j,3*k)=(v.slave[k]*v.h+v.normal_weights[k]*f.guard.normal).transpose();
    rows.block<1,3>(j,9+3*k)=(-v.main[k]*v.h).transpose();
   }
  }return rows;
 }
 // Jacobian-vector product of every scalar gap; never constructs dormant guard rows.
 py::tuple directional(const Eigen::MatrixXd&ds,const Eigen::MatrixXd&dm,const Eigen::VectorXi&unbuilt)const{
  int n=(int)faces.size();if(ds.rows()!=n||dm.rows()!=n||ds.cols()!=9||dm.cols()!=9)throw std::invalid_argument("Invalid directional increments");
  Eigen::VectorXd g(12*n);int row=0;
  for(int i=0;i<n;i++){
   const auto&f=faces[i];Tri u,v;Eigen::Vector3d dn;
   for(int k=0;k<3;k++){u[k]=ds.block<1,3>(i,3*k).transpose();v[k]=dm.block<1,3>(i,3*k).transpose();dn[k]=f.guard.normal.dot(u[k]);}
   for(const auto&z:f.vertices){Eigen::Vector3d w=Eigen::Vector3d::Zero();for(int k=0;k<3;k++)w+=z.slave[k]*u[k]-z.main[k]*v[k];
    g[row++]=z.h.dot(w)+z.normal_weights.dot(dn);}
  }
  Eigen::MatrixXd weak(unbuilt.size(),6);
  for(int j=0;j<unbuilt.size();j++){
   int i=unbuilt[j];const auto&f=faces.at(i);std::array<V3<D1>,3>s,m;
   for(int k=0;k<9;k++){s[k/3][k%3]=D1(f.s[k/3][k%3]);m[k/3][k%3]=D1(f.m[k/3][k%3]);s[k/3][k%3].derivatives()[0]=ds(i,k);m[k/3][k%3].derivatives()[0]=dm(i,k);}
   auto c=integrate(s,m,band,false,nullptr,retain_negative?wide:-1.);for(int k=0;k<3;k++){weak(j,k)=c.area[k].derivatives()[0];weak(j,k+3)=c.gap[k].derivatives()[0];}
  }
  g.conservativeResize(row);return py::make_tuple(g,weak);
 }
 Eigen::MatrixXd pressure_partials(const Eigen::VectorXi&pairs)const{
  Eigen::MatrixXd out(pairs.size(),108);
  for(int i=0;i<pairs.size();i++){
   const auto&f=faces.at(pairs[i]);std::array<V3<D>,3>s,m;
   for(int k=0;k<9;k++){s[k/3][k%3]=D(f.s[k/3][k%3]);m[k/3][k%3]=D(f.m[k/3][k%3]);s[k/3][k%3].derivatives()[k]=1;m[k/3][k%3].derivatives()[k+9]=1;}
   auto c=integrate(s,m,band,false,nullptr,retain_negative?wide:-1.);
   for(int k=0;k<3;k++){out.block<1,18>(i,k*18)=c.area[k].derivatives().transpose();out.block<1,18>(i,(k+3)*18)=c.gap[k].derivatives().transpose();}
  }return out;
 }
};
inline void bind_shared_overlap(py::module_&m){
 py::class_<SharedFaceBatch>(m,"SharedFaceBatch").def(py::init<const Eigen::MatrixXd&,const Eigen::MatrixXd&,double,double,bool,bool>(),py::arg("secondary"),py::arg("main"),py::arg("radius"),py::arg("band"),py::arg("linearize"),py::arg("asymmetric")=false)
 .def("values",&SharedFaceBatch::values).def("vertex_rows",&SharedFaceBatch::vertex_rows)
 .def("directional",&SharedFaceBatch::directional).def("pressure_partials",&SharedFaceBatch::pressure_partials)
 .def_readonly("common_prepares",&SharedFaceBatch::common_prepares).def_readonly("near_tests",&SharedFaceBatch::near_tests);
}
