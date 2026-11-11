// Appended to overlap.cu: identical finite-overlap values, fused median recovery.
__device__ void min_double(double*address,double v){
 unsigned long long* p=(unsigned long long*)address,old=*p,assumed;
 do{assumed=old;if(__longlong_as_double(old)<=v)break;
 old=atomicCAS(p,assumed,__double_as_longlong(v));}while(old!=assumed);
}
__device__ V<double,3> read_point(const double*x,int id){V<double,3>o;for(int k=0;k<3;k++)o[k]=x[3*id+k];return o;}
__device__ V<double,3> bary3(V<double,3>p,V<double,3>*t){
 auto u=t[1]-t[0],v=t[2]-t[0],d=p-t[0];double uu=dot(u,u),uv=dot(u,v),vv=dot(v,v),den=uu*vv-uv*uv;
 double b=(vv*dot(d,u)-uv*dot(d,v))/den,c=(uu*dot(d,v)-uv*dot(d,u))/den;
 V<double,3>o;o[0]=1-b-c;o[1]=b;o[2]=c;return o;
}
extern "C" __global__ void nodal_normals(const double*x,const int*faces,double*out,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 auto a=read_point(x,faces[3*i]),b=read_point(x,faces[3*i+1]),c=read_point(x,faces[3*i+2]);auto v=cross(b-a,c-a);
 for(int j=0;j<3;j++)for(int k=0;k<3;k++)atomicAdd(out+3*faces[3*i+j]+k,v[k]);
}
extern "C" __global__ void normal_patches(const double*sm,const double*mm,const int*fs,const int*fm,
 const int*owners,const int*mids,const double*normals,int count,double radius,double band,
 double*area,double*integral,int*counts,double*covered,double*reference,int nf,
 double*piece_values,int*piece_nodes,int*piece_row,double*minimum,int*error){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=6*count)return;
 piece_row[id]=-1;int p=id/6,piece=id%6,vertex=piece/2,side=piece%2,face=owners[p],master=mids[p];
 V<double,3>s[3],m[3];for(int k=0;k<3;k++){s[k]=read_point(sm,fs[3*face+k]);m[k]=read_point(mm,fm[3*master+k]);}
 auto outward=read_point(normals,fs[3*face+vertex]);double length=norm(outward);
 if(length<1e-20){atomicOr(error,1);return;}outward=outward/length;
 V<double,3>real[3],embedded[3];double height[3];auto center=(s[0]+s[1]+s[2])/3.;real[0]=s[vertex];
 if(!side){real[1]=(s[vertex]+s[(vertex+1)%3])/2.;real[2]=center;}
 else{real[1]=center;real[2]=(s[vertex]+s[(vertex+2)%3])/2.;}
 for(int k=0;k<3;k++){double d=dot(real[k]-s[vertex],outward);embedded[k]=real[k]-outward*d;height[k]=-d;}
 auto cr=cross(real[1]-real[0],real[2]-real[0]);auto projected=cross(embedded[1]-embedded[0],embedded[2]-embedded[0]);
 if(dot(cr,outward)<=1e-20){atomicOr(error,2);return;}
 double scale=norm(cr)/norm(projected),physical_area=norm(cr)/2.;int virtual_id=6*face+piece;
 atomicExch((unsigned long long*)(reference+virtual_id),__double_as_longlong(physical_area));
 Frame<double>f;if(!f.prepare(embedded,m))return;
 auto guard=evaluate(f,embedded,m,radius,true,-1.);if(!guard.samples)return;
 auto patch=evaluate(f,embedded,m,band,false,radius);
 double guard_area=guard.area[0]+guard.area[1]+guard.area[2];atomicAdd(covered+virtual_id,guard_area*scale);
 for(int j=0;j<guard.count;j++){
  V<double,3>pt;for(int k=0;k<3;k++)pt[k]=guard.vp[3*j+k];auto bc=bary3(pt,embedded);
  double gap=guard.vg[j];for(int k=0;k<3;k++)gap+=bc[k]*height[k];min_double(minimum,gap);
  if(gap<-radius)atomicOr(error,4);
 }
 if(!patch.samples)return;
 double a=0.,I=0.;V<double,3>point,normal;for(int k=0;k<3;k++)normal[k]=patch.normal[k];
 for(int j=0;j<3;j++){
  V<double,3>c;for(int k=0;k<3;k++)c[k]=(patch.points[3*j+k]+patch.gap[j]*normal[k])/fmax(patch.area[j],1e-300);
  auto bc=bary3(c,embedded);double hi=0.;for(int k=0;k<3;k++)hi+=bc[k]*height[k];
  a+=patch.area[j];I+=patch.gap[j]+patch.area[j]*hi;
  for(int k=0;k<3;k++)point[k]+=patch.points[3*j+k];
 }
 atomicAdd(covered+6*nf+virtual_id,a*scale);
 double g=I/a;point=point/a;auto bs=bary3(point+normal*g,s),bm=bary3(point,m);
 int row=fs[3*face+vertex];atomicAdd(area+row,a*scale);atomicAdd(integral+row,I*scale);atomicAdd(counts+row,1);piece_row[id]=row;
 for(int j=0;j<6;j++){
  piece_nodes[6*id+j]=j<3?fs[3*face+j]:fm[3*master+j-3];
  double weight=(j<3?bs[j]:-bm[j-3])*a*scale;
  for(int k=0;k<3;k++)piece_values[18*id+3*j+k]=weight*normal[k];
 }
}
extern "C" __global__ void check_normal_coverage(const double*covered,const double*reference,int nf,int*error){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=6*nf)return;
 if(covered[i]>reference[i]*(1+1e-7)||covered[6*nf+i]>reference[i]*(1+1e-7))atomicOr(error,8);
}
extern "C" __global__ void pack_normal_rows(const int*piece_row,const int*piece_nodes,const double*piece_values,
 const double*area,const int*lut,const int*ptr,int*cursor,int*cols,double*values,int np,int slave_offset,int master_offset){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=np)return;int raw=piece_row[i];if(raw<0)return;
 int row=lut[raw];if(row<0)return;int begin=ptr[row]+18*atomicAdd(cursor+row,1);
 for(int j=0;j<6;j++)for(int k=0;k<3;k++){
  int at=begin+3*j+k;cols[at]=3*(piece_nodes[6*i+j]+(j<3?slave_offset:master_offset))+k;
  values[at]=piece_values[18*i+3*j+k]/area[raw];
 }
}
