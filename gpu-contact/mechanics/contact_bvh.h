// Persistent AABB hierarchy for conservative triangle-box queries.
// Refitting changes bounds only. Sorted leaf IDs preserve assembly order.
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>
#include <algorithm>
#include <numeric>
#include <vector>
#include <stdexcept>
namespace py=pybind11;
class ContactAABBTree {
    struct Node { Eigen::Vector3d lo,hi; int begin,end,left=-1,right=-1; };
    std::vector<Node> nodes;
    std::vector<int> order;
    Eigen::MatrixXd low,high;
    int leaf_size;
    void bounds(int id) {
        auto& n=nodes[id];n.lo.setConstant(INFINITY);n.hi.setConstant(-INFINITY);
        if(n.left<0)for(int j=n.begin;j<n.end;j++) {
            n.lo=n.lo.cwiseMin(low.row(order[j]).transpose());
            n.hi=n.hi.cwiseMax(high.row(order[j]).transpose());
        } else {
            n.lo=nodes[n.left].lo.cwiseMin(nodes[n.right].lo);
            n.hi=nodes[n.left].hi.cwiseMax(nodes[n.right].hi);
        }
    }
    int build(int begin,int end) {
        int id=int(nodes.size());nodes.push_back(Node{});
        nodes[id].begin=begin;nodes[id].end=end;
        bounds(id);
        if(end-begin>leaf_size) {
            Eigen::Index axis;(nodes[id].hi-nodes[id].lo).maxCoeff(&axis);
            int mid=(begin+end)/2;
            std::nth_element(order.begin()+begin,order.begin()+mid,order.begin()+end,[&](int a,int b){
                double ca=low(a,axis)+high(a,axis),cb=low(b,axis)+high(b,axis);
                return ca==cb?a<b:ca<cb;
            });
            int left=build(begin,mid),right=build(mid,end);
            nodes[id].left=left;nodes[id].right=right;
        }
        return id;
    }
    static void validate(const Eigen::MatrixXd& a,const Eigen::MatrixXd& b) {
        if(a.cols()!=3||b.cols()!=3||a.rows()!=b.rows()||!a.allFinite()||!b.allFinite()||(a.array()>b.array()).any())
            throw std::invalid_argument("Expected finite matching N x 3 AABBs");
    }
public:
    ContactAABBTree(const Eigen::MatrixXd& a,const Eigen::MatrixXd& b,int leaf=8):low(a),high(b),leaf_size(leaf) {
        validate(a,b);if(leaf<1)throw std::invalid_argument("Positive leaf size required");
        order.resize(a.rows());std::iota(order.begin(),order.end(),0);
        nodes.reserve(2*order.size());if(!order.empty())build(0,int(order.size()));
    }
    void refit(const Eigen::MatrixXd& a,const Eigen::MatrixXd& b) {
        validate(a,b);if(a.rows()!=low.rows())throw std::invalid_argument("BVH topology changed");
        low=a;high=b;for(int i=int(nodes.size())-1;i>=0;i--)bounds(i);
    }
    size_t bytes() const {return nodes.capacity()*sizeof(Node)+order.capacity()*sizeof(int)+(low.size()+high.size())*sizeof(double);}
    py::dict query(const Eigen::MatrixXd& qlow,const Eigen::MatrixXd& qhigh,long long max_pairs) const {
        validate(qlow,qhigh);if(max_pairs<0)throw std::invalid_argument("Negative pair budget");
        std::vector<int> ids,stack;std::vector<long long> offsets(qlow.rows()+1,0);bool complete=true;
        {
        py::gil_scoped_release release;
        for(int i=0;i<qlow.rows();i++) {
            Eigen::Vector3d a=qlow.row(i),b=qhigh.row(i);stack.clear();if(!nodes.empty())stack.push_back(0);
            size_t begin=ids.size();
            while(!stack.empty()) {
                int j=stack.back();stack.pop_back();const auto& n=nodes[j];
                if((n.hi.array()<a.array()).any()||(n.lo.array()>b.array()).any())continue;
                if(n.left>=0){stack.push_back(n.right);stack.push_back(n.left);continue;}
                for(int k=n.begin;k<n.end;k++) {
                    int f=order[k];
                    if((high.row(f).transpose().array()<a.array()).any()||(low.row(f).transpose().array()>b.array()).any())continue;
                    if(ids.size()>=size_t(max_pairs)){complete=false;break;}
                    ids.push_back(f);
                }
                if(!complete)break;
            }
            if(!complete)break;
            std::sort(ids.begin()+begin,ids.end());offsets[i+1]=static_cast<long long>(ids.size());
        }
        }
        py::dict result;result["complete"]=complete;
        if(complete) {
            Eigen::VectorXi index(ids.size());for(size_t i=0;i<ids.size();i++)index[i]=ids[i];
            Eigen::Matrix<long long,Eigen::Dynamic,1> cuts(offsets.size());for(size_t i=0;i<offsets.size();i++)cuts[i]=offsets[i];
            result["indices"]=index;result["offsets"]=cuts;
        }
        return result;
    }
};
