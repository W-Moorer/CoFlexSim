"""CUDA counterpart of geometry-only AnchoredP1PressureContact.

Clipping is evaluated once. Store each overlap's exact integrals, select the
closest opposing smooth feature, then accumulate only that feature's pieces.
No reference trajectory or force enters this choice.
"""
EXTRA=r'''
__device__ double anchor_distance2(V<double,3>p,V<double,3>*m){
 auto bc=bary3(p,m);double best=1e300;
 if(bc[0]>=0&&bc[1]>=0&&bc[2]>=0){auto q=m[0]*bc[0]+m[1]*bc[1]+m[2]*bc[2];best=dot(p-q,p-q);}
 for(int j=0;j<3;j++){auto e=m[(j+1)%3]-m[j];double w=fmin(1.,fmax(0.,dot(p-m[j],e)/dot(e,e)));auto d=p-(m[j]+e*w);best=fmin(best,dot(d,d));}
 return best;
}
extern "C" __global__ void choose_feature(const int*rows,const double*distance,const int*feature,const double*best,int*chosen,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||rows[i]<0)return;
 int r=rows[i];if(distance[i]==best[r])atomicMin(chosen+r,feature[i]);
}
extern "C" __global__ void accumulate_feature(int*rows,const int*feature,const int*chosen,const double*piece,
 double*area,double*integral,double*stiffness,int*counts,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||rows[i]<0)return;int r=rows[i];
 if(feature[i]!=chosen[r]){rows[i]=-1;return;}
 atomicAdd(area+r,piece[3*i]);atomicAdd(integral+r,piece[3*i+1]);atomicAdd(stiffness+r,piece[3*i+2]);atomicAdd(counts+r,1);
}
'''

def transform(src):
    marker='double*stiffness){'
    assert src.count(marker)==1
    src=src.replace(marker,'double*stiffness,const int*primary_features,double*piece_integrals,double*piece_distances,int*piece_features,double*best){')
    src=src.replace('if(face<0||master<0)return;','if(face<0||master<0||side)return;')
    old='''if(!side){real[1]=(s[vertex]+s[(vertex+1)%3])/2.;real[2]=center;}
 else{real[1]=center;real[2]=(s[vertex]+s[(vertex+2)%3])/2.;}'''
    assert old in src;src=src.replace(old,'for(int k=0;k<3;k++)real[k]=s[k];')
    marker='V<double,3>c;for(int k=0;k<3;k++)c[k]=(patch.points'
    assert marker in src;src=src.replace(marker,'if(j!=vertex)continue;\n  '+marker)
    src=src.replace('double g=I/a;','if(a<=1e-24)return;double g=I/a;')
    old='''int row=features[3*face+vertex];atomicAdd(stiffness+row,a*scale*face_k[face]);atomicAdd(area+row,a*scale);atomicAdd(integral+row,I*scale);atomicAdd(counts+row,1);piece_row[id]=row;'''
    assert old in src
    src=src.replace(old,'''int row=features[3*face+vertex];piece_row[id]=row;
 piece_integrals[3*id]=a*scale;piece_integrals[3*id+1]=I*scale;piece_integrals[3*id+2]=a*scale*face_k[face];
 double distance=anchor_distance2(s[vertex],m);piece_distances[id]=distance;piece_features[id]=primary_features[master];min_double(best+row,distance);''')
    # distance helper must appear after vector helpers, before normal_patches.
    at=src.index('extern "C" __global__ void normal_patches')
    return src[:at]+EXTRA+src[at:]
