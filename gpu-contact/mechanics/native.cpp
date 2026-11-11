// CoFlex-PD native tetrahedral projective energy. Original author code is untouched.
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <stdexcept>
#include <string>
#include <cstring>
#include <mutex>
#include <cstdint>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifndef COFLEX_MODULE_NAME
#define COFLEX_MODULE_NAME coflex_native
#endif
namespace py=pybind11;
using namespace Eigen;
using Sp=SparseMatrix<double>;
struct Tet { Vector4i ids; Matrix<double,4,3> G; double V; };

// Closest positive singular values with determinant one. The first release is
// restricted to positive, modest elastic strains; a failed local solve is fatal.
Vector3d volume_projection(const Vector3d& s,bool small_schur=false) {
    Vector3d p=s/std::cbrt(s.prod()); double eta=0;
    for(int it=0;it<50;++it) {
        Vector3d c(p[1]*p[2],p[0]*p[2],p[0]*p[1]);
        Vector4d r; r.head<3>()=p-s+eta*c; r[3]=p.prod()-1;
        if(r.norm()<2e-12) return p;
        Matrix4d A=Matrix4d::Zero(); A.topLeftCorner<3,3>().setIdentity();
        A(0,1)=A(1,0)=eta*p[2]; A(0,2)=A(2,0)=eta*p[1]; A(1,2)=A(2,1)=eta*p[0];
        A.topRightCorner<3,1>()=c; A.bottomLeftCorner<1,3>()=c.transpose();
        Vector4d d;
        // In the diagonally dominant positive branch, eliminate the one
        // multiplier through the 3x3 Schur complement. Outside this safe
        // range retain the pivoted reference solve. Same Newton equations,
        // residual tolerance and positive-strain line search in either path.
        const double h01=eta*p[2],h02=eta*p[1],h12=eta*p[0];
        if(small_schur && std::max({std::abs(h01),std::abs(h02),std::abs(h12)})<0.25){
            Matrix3d adj;
            adj << 1-h12*h12,h02*h12-h01,h01*h12-h02,
                   h02*h12-h01,1-h02*h02,h01*h02-h12,
                   h01*h12-h02,h01*h02-h12,1-h01*h01;
            const double det=1+2*h01*h02*h12-h01*h01-h02*h02-h12*h12;
            const Vector3d u=adj*(-r.head<3>())/det,v=adj*c/det;
            d[3]=(r[3]+c.dot(u))/c.dot(v);d.head<3>()=u-v*d[3];
        }else d=A.fullPivLu().solve(-r);
        double a=1; bool accepted=false;
        for(int ls=0;ls<30;++ls,a*=0.5) {
            Vector3d pn=p+a*d.head<3>(); double en=eta+a*d[3];
            if(pn.minCoeff()<=0)continue;
            Vector3d cn(pn[1]*pn[2],pn[0]*pn[2],pn[0]*pn[1]);
            Vector4d rn; rn.head<3>()=pn-s+en*cn; rn[3]=pn.prod()-1;
            if(rn.norm()<r.norm()){p=pn;eta=en;accepted=true;break;}
        }
        if(!accepted)break;
    }
    throw std::runtime_error("Volume projection did not converge in the positive-strain branch");
}

class TetModel {
public:
    MatrixXd X; MatrixXi T; std::vector<Tet> elements; VectorXd mass; Sp L,K;
    struct ElementProjection {
        Matrix3d F, P;
        double energy=0., smallest=0.;
        bool valid=false;
        std::uint64_t hits=0;
#ifdef COFLEX_AUDIT_PROJECTION_OUTPUTS
        Matrix3d rotation,volume;
        std::uint64_t same_rotation=0,same_volume=0,same_both=0;
#endif
    };
    mutable std::vector<ElementProjection> previous;
    mutable std::mutex cache_mutex;
    mutable std::uint64_t cache_visits=0;
    py::dict element_cache_report() const {
        std::lock_guard<std::mutex> lock(cache_mutex);
        std::vector<std::uint64_t> hits;
        for(const auto& entry:previous)hits.push_back(entry.hits);
        py::dict r;r["visits_per_element"]=cache_visits;r["hits"]=py::cast(hits);
        r["allocated_bytes"]=previous.capacity()*sizeof(ElementProjection);
#ifdef COFLEX_AUDIT_PROJECTION_OUTPUTS
        std::uint64_t rotation=0,volume=0,both=0;
        std::vector<int> ids;
        for(size_t i=0;i<previous.size();++i){const auto& entry=previous[i];
            rotation+=entry.same_rotation;volume+=entry.same_volume;both+=entry.same_both;
            if(entry.same_both)ids.push_back(int(i));
        }
        r["changed_F_same_rotation"]=rotation;r["changed_F_same_volume"]=volume;
        r["changed_F_same_both"]=both;r["changed_F_same_both_ids"]=py::cast(ids);
#endif
        return r;
    }
    double mu,lambda;
    TetModel(const MatrixXd& x,const MatrixXi& t,double E,double nu,double rho,bool assemble=true):X(x),T(t) {
        if(x.cols()!=3||t.cols()!=4||E<=0||nu<0||nu>=0.49||rho<=0)throw std::invalid_argument("Invalid tetrahedral material or mesh");
        mu=E/(2*(1+nu));lambda=E*nu/((1+nu)*(1-2*nu));
        mass=VectorXd::Zero(X.rows()); std::vector<Triplet<double>> l,k;
        // Exact known triplet counts avoid geometric capacity growth at setup.
        if(assemble){l.reserve(size_t(T.rows())*48);k.reserve(size_t(T.rows())*144);}
        for(int e=0;e<T.rows();++e){
            Tet a; a.ids=T.row(e); Matrix3d D;
            for(int j=0;j<4;++j)if(a.ids[j]<0||a.ids[j]>=X.rows())throw std::invalid_argument("Invalid vertex index");
            for(int j=0;j<3;++j)D.col(j)=(X.row(a.ids[j+1])-X.row(a.ids[0])).transpose();
            a.V=std::abs(D.determinant())/6;
            if(a.V<1e-22)throw std::invalid_argument("Degenerate tetrahedron");
            a.G.bottomRows<3>()=D.inverse(); a.G.row(0)=-a.G.bottomRows<3>().colwise().sum();
            for(int i=0;i<4;++i)mass[a.ids[i]]+=rho*a.V/4;
            if(assemble)for(int i=0;i<4;++i)for(int j=0;j<4;++j){
                Vector3d gi=a.G.row(i),gj=a.G.row(j);
                Matrix3d kij=a.V*(mu*gi.dot(gj)*Matrix3d::Identity()+mu*gj*gi.transpose()+lambda*gi*gj.transpose());
                double lij=a.V*(2*mu+3*lambda)*gi.dot(gj);
                for(int d=0;d<3;++d){l.emplace_back(3*a.ids[i]+d,3*a.ids[j]+d,lij);
                    for(int c=0;c<3;++c)k.emplace_back(3*a.ids[i]+d,3*a.ids[j]+c,kij(d,c));}
            }
            elements.push_back(a);
        }
        L.resize(3*X.rows(),3*X.rows());K.resize(3*X.rows(),3*X.rows());
        L.setFromTriplets(l.begin(),l.end());K.setFromTriplets(k.begin(),k.end());
        if(mass.minCoeff()<=0)throw std::invalid_argument("Unused vertex in mesh");
    }
    size_t assembled_operator_bytes() const {
        auto bytes=[](const Sp& A)->size_t{return A.cols()?size_t(A.nonZeros())*(sizeof(double)+sizeof(Sp::StorageIndex))+size_t(A.outerSize()+1)*sizeof(Sp::StorageIndex):0;};
        return bytes(L)+bytes(K);
    }
    size_t release_assembled_operators() {
        // Element geometry, mass and constitutive data remain valid. This is
        // called only after the solver owns its projected operators/factors.
        size_t bytes=assembled_operator_bytes();Sp emptyL,emptyK;L.swap(emptyL);K.swap(emptyK);return bytes;
    }
    Matrix3d F_of(const MatrixXd& x,const Tet& a) const {
        Matrix<double,3,4> xe; for(int i=0;i<4;++i)xe.col(i)=x.row(a.ids[i]); return xe*a.G;
    }
    py::dict evaluate(const MatrixXd& x,bool fields=false) const {
        if(x.rows()!=X.rows()||x.cols()!=3)throw std::invalid_argument("Wrong position dimensions");
        MatrixXd g=MatrixXd::Zero(X.rows(),3),stress(fields?T.rows():0,9),strain(fields?T.rows():0,9);
        double energy=0,mindet=1e100,maxdet=-1e100;
        for(size_t ei=0;ei<elements.size();++ei){const auto& a=elements[ei];Matrix3d F=F_of(x,a);double det=F.determinant();
            if(det<=1e-10)throw std::runtime_error("Inverted or singular tetrahedron");
            mindet=std::min(mindet,det);maxdet=std::max(maxdet,det);
            JacobiSVD<Matrix3d> svd(F,ComputeFullU|ComputeFullV); Matrix3d R=svd.matrixU()*svd.matrixV().transpose();
            Matrix3d Q=svd.matrixU()*volume_projection(svd.singularValues()).asDiagonal()*svd.matrixV().transpose();
            Matrix3d P=2*mu*(F-R)+3*lambda*(F-Q);
            energy+=a.V*(mu*(F-R).squaredNorm()+1.5*lambda*(F-Q).squaredNorm());
            for(int i=0;i<4;++i)g.row(a.ids[i])+=(a.V*P*a.G.row(i).transpose()).transpose();
            if(fields){Matrix3d sigma=P*F.transpose()/det,eps=0.5*(F.transpose()*F-Matrix3d::Identity());
                for(int i=0;i<3;++i)for(int j=0;j<3;++j){stress(ei,3*i+j)=sigma(i,j);strain(ei,3*i+j)=eps(i,j);}}
        }
        py::dict r;r["energy"]=energy;r["gradient"]=g;r["min_det"]=mindet;r["max_det"]=maxdet;
        if(fields){r["stress"]=stress;r["strain"]=strain;}return r;
    }
    double inversion_step(const MatrixXd& x,const MatrixXd& dx) const {
        double amax=1;
        for(const auto& a:elements){Matrix3d F=F_of(x,a),dF=F_of(dx,a);
            double norm=dF.norm(); if(norm==0)continue;
            double sigma=JacobiSVD<Matrix3d>(F).singularValues().minCoeff();
            amax=std::min(amax,0.8*sigma/norm);
        }return amax;
    }
    double inversion_step_from_singular_values(const MatrixXd& dx,const VectorXd& sigma) const {
        if(dx.rows()!=X.rows()||dx.cols()!=3||sigma.size()!=static_cast<int>(elements.size())||
           !sigma.allFinite()||sigma.minCoeff()<=0)throw std::invalid_argument("Invalid cached element singular values");
        double amax=1;
        py::gil_scoped_release release;
        for(size_t ei=0;ei<elements.size();++ei){
            double norm=F_of(dx,elements[ei]).norm();
            if(norm>0)amax=std::min(amax,0.8*sigma[ei]/norm);
        }
        return amax;
    }
    py::dict evaluate_parallel(const MatrixXd& x,bool fields=false,int threads=1,bool element_cache=false) const {
        if(x.rows()!=X.rows()||x.cols()!=3||threads<1)throw std::invalid_argument("Invalid parallel projection input");
        const int ne=static_cast<int>(elements.size());
        MatrixXd gradients(ne,12),stress(fields?ne:0,9),strain(fields?ne:0,9);
        VectorXd energies(ne),dets(ne),smallest(ne);std::vector<std::string> errors(ne);
        MatrixXd g=MatrixXd::Zero(X.rows(),3);double energy=0,mindet=1e100,maxdet=-1e100;
        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> lock(cache_mutex,std::defer_lock);
            const bool reuse=element_cache && !fields;
            if(reuse){
                lock.lock();
                previous.resize(ne);
                ++cache_visits;
            }
            // Disjoint element buffers; deterministic serial scatter avoids races
            // and makes results independent of the OpenMP reduction order.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if(threads>1)
#endif
            for(int ei=0;ei<ne;++ei){
                try {
                    const auto& a=elements[ei];Matrix3d F=F_of(x,a);double det=F.determinant();
                    if(det<=1e-10)throw std::runtime_error("Inverted or singular tetrahedron");
                    ElementProjection* entry=reuse?&previous[ei]:nullptr;
                    Matrix3d P;
                    if(entry && entry->valid && std::memcmp(F.data(),entry->F.data(),9*sizeof(double))==0){
                        ++entry->hits;
                        P=entry->P;energies[ei]=entry->energy;smallest[ei]=entry->smallest;
                    }else{
                        JacobiSVD<Matrix3d> svd(F,ComputeFullU|ComputeFullV);
                        smallest[ei]=svd.singularValues().minCoeff();
                        Matrix3d R=svd.matrixU()*svd.matrixV().transpose();
                        Matrix3d Q=svd.matrixU()*volume_projection(svd.singularValues(),true).asDiagonal()*svd.matrixV().transpose();
                        P=2*mu*(F-R)+3*lambda*(F-Q);
                        energies[ei]=a.V*(mu*(F-R).squaredNorm()+1.5*lambda*(F-Q).squaredNorm());
#ifdef COFLEX_AUDIT_PROJECTION_OUTPUTS
                        // Retrospective equality of the computed outputs, never an online hit test.
                        if(entry){
                            if(entry->valid){
                                bool same_r=std::memcmp(R.data(),entry->rotation.data(),9*sizeof(double))==0;
                                bool same_q=std::memcmp(Q.data(),entry->volume.data(),9*sizeof(double))==0;
                                entry->same_rotation+=same_r;entry->same_volume+=same_q;entry->same_both+=same_r && same_q;
                            }
                            entry->rotation=R;entry->volume=Q;
                        }
#endif
                        if(entry){entry->F=F;entry->P=P;entry->energy=energies[ei];entry->smallest=smallest[ei];entry->valid=true;}
                    }
                    dets[ei]=det;
                    for(int i=0;i<4;++i)gradients.block<1,3>(ei,3*i)=(a.V*P*a.G.row(i).transpose()).transpose();
                    if(fields){Matrix3d sigma=P*F.transpose()/det,eps=.5*(F.transpose()*F-Matrix3d::Identity());
                        for(int i=0;i<3;++i)for(int j=0;j<3;++j){stress(ei,3*i+j)=sigma(i,j);strain(ei,3*i+j)=eps(i,j);}}
                } catch(const std::exception& e){errors[ei]=e.what();}
            }
            for(int ei=0;ei<ne;++ei){
                if(!errors[ei].empty())throw std::runtime_error(errors[ei]);
                energy+=energies[ei];mindet=std::min(mindet,dets[ei]);maxdet=std::max(maxdet,dets[ei]);
                for(int i=0;i<4;++i)g.row(elements[ei].ids[i])+=gradients.block<1,3>(ei,3*i);
            }
        }
        py::dict r;r["energy"]=energy;r["gradient"]=g;r["min_det"]=mindet;r["max_det"]=maxdet;r["threads"]=threads;r["minimum_singular_values"]=smallest;
        if(fields){r["stress"]=stress;r["strain"]=strain;}return r;
    }
};
PYBIND11_MODULE(COFLEX_MODULE_NAME,m){
    m.attr("energy_id")="tet_so3_det1_pd_v1";
#ifdef _OPENMP
    m.attr("openmp_enabled")=true;
#else
    m.attr("openmp_enabled")=false;
#endif
    py::class_<TetModel>(m,"TetModel",py::module_local())
      .def(py::init<const MatrixXd&,const MatrixXi&,double,double,double,bool>(),py::arg("X"),py::arg("T"),py::arg("young"),py::arg("poisson"),py::arg("density"),py::arg("assemble")=true)
      .def_readonly("mass",&TetModel::mass).def_readonly("L",&TetModel::L).def_readonly("K",&TetModel::K)
      .def_property_readonly("assembled_operator_bytes",&TetModel::assembled_operator_bytes)
      .def("release_assembled_operators",&TetModel::release_assembled_operators)
      .def("evaluate",&TetModel::evaluate,py::arg("positions"),py::arg("fields")=false)
      .def("evaluate_parallel",&TetModel::evaluate_parallel,py::arg("positions"),py::arg("fields")=false,py::arg("threads")=1,py::arg("element_cache")=false)
      .def("element_cache_report",&TetModel::element_cache_report)
      .def("inversion_step",&TetModel::inversion_step)
      .def("inversion_step_from_singular_values",&TetModel::inversion_step_from_singular_values);
}
