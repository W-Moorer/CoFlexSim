"""Validated GPU replay of polygon-clipping topology with current coordinates.

Every replay recomputes all current half-plane signs. A changed sign pattern
uses the original clip and records fresh instructions. No old geometry values
or old integration weights are reused.
"""
REPLAY=r'''
struct ClipReplay {unsigned long long*cache;unsigned long long*hits;unsigned long long*misses;int slot;};
template<class T> __device__ Poly<T> clip_replay(Poly<T>p,V<T,3>c,ClipReplay*ctx){
 if(!ctx)return clip(p,c);int slot=ctx->slot++;Poly<T>o;if(p.count<3)return o;
 T f[12];unsigned int mask=0;for(int k=0;k<p.count;k++){
  f[k]=p.p[k][0]*c[0]+p.p[k][1]*c[1]+c[2]+T(2e-13);if(value(f[k])>=0)mask|=1u<<k;}
 unsigned long long signature=(1ull<<63)|((unsigned long long)p.count<<16)|mask;
 unsigned long long old=ctx->cache[2*slot],ops=ctx->cache[2*slot+1];
 if((old&~(15ull<<24))==signature){
  o.count=(old>>24)&15;for(int j=0;j<o.count;j++){int code=(ops>>(5*j))&31,k=code&15;
   if(code&16){int prev=k?k-1:p.count-1;o.p[j]=p.p[prev]+(p.p[k]-p.p[prev])*(f[prev]/(f[prev]-f[k]));}
   else o.p[j]=p.p[k];}
  (*ctx->hits)++;return o;
 }
 (*ctx->misses)++;ops=0;int prev=p.count-1;
 for(int k=0;k<p.count;k++){
  if((value(f[prev])>=0)!=(value(f[k])>=0)){
   if(o.count>=12){o.count=-1;return o;}ops|=(unsigned long long)(16+k)<<(5*o.count);
   o.p[o.count++]=p.p[prev]+(p.p[k]-p.p[prev])*(f[prev]/(f[prev]-f[k]));}
  if(value(f[k])>=0){if(o.count>=12){o.count=-1;return o;}ops|=(unsigned long long)k<<(5*o.count);o.p[o.count++]=p.p[k];}
  prev=k;
 }
 if(o.count<3)o.count=0;
 ctx->cache[2*slot]=signature|((unsigned long long)o.count<<24);ctx->cache[2*slot+1]=ops;return o;
}
'''

def overlap_source(source):
    marker='template<class T> struct Frame {'
    assert source.count(marker)==1
    source=source.replace(marker,REPLAY+'\n'+marker)
    source=source.replace('bool prepare(V<T,3>*s,V<T,3>*m){','bool prepare(V<T,3>*s,V<T,3>*m,ClipReplay*ctx=nullptr){')
    source=source.replace('poly=clip(poly,c);','poly=clip_replay(poly,c,ctx);')
    source=source.replace('double negative){','double negative,ClipReplay*ctx=nullptr){')
    return source

def patch_source(source):
    sig='const int*features,const double*face_k,double*stiffness)'
    assert source.count(sig)==1
    source=source.replace(sig,sig[:-1]+',unsigned long long*cache,unsigned long long*hits,unsigned long long*misses)')
    source=source.replace('Frame<double>f;if(!f.prepare(embedded,m))return;',
        'ClipReplay ctx{cache+14ull*id,hits+id,misses+id,0}; Frame<double>f;if(!f.prepare(embedded,m,&ctx))return;')
    source=source.replace('auto guard=evaluate(f,embedded,m,radius,true,-1.);',
        'ctx.slot=3;auto guard=evaluate(f,embedded,m,radius,true,-1.,&ctx);')
    source=source.replace('auto patch=evaluate(f,embedded,m,band,false,radius);',
        'ctx.slot=5;auto patch=evaluate(f,embedded,m,band,false,radius,&ctx);')
    return source
