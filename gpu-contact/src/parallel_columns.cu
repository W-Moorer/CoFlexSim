// Launch by surface/contact row, rather than worst-case rows*capacity. This
// avoids thousands of empty blocks in batched, device-guarded graph replay.
extern "C" __global__ void response_rows(const double*C,const int*ptr,const int*col,const double*val,
 const long long*ids,const int*meta,double*T,int p,int cap,int column_major){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(meta[4]||meta[5]||i>=p)return;
 for(int j=meta[3];j<meta[2];j++){
  int row=ids[j];double v=0.;
  for(int e=ptr[row];e<ptr[row+1];e++)if(val[e]!=0.)
   v+=C[column_major?(long long)col[e]*p+i:(long long)i*p+col[e]]*val[e];
  T[i*cap+j]=v;
 }
}
extern "C" __global__ void contact_rows(const int*ptr,const int*col,const double*val,const double*v,
 const long long*ids,const int*meta,const double*T,double*W,int cap,int add){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(meta[4]||meta[5]||i>=meta[0])return;
 for(int j=meta[3];j<meta[2];j++){
  double s=v[i]*v[ids[j]];
  for(int e=ptr[i];e<ptr[i+1];e++)if(val[e]!=0.)s+=val[e]*T[col[e]*cap+j];
  if(add)W[i*cap+j]+=s;else W[i*cap+j]=s;
 }
}
