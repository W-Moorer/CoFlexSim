#include "coflex_native.h"
#include "internal.h"
#include <cublas_v2.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <memory>
#include <vector>
#include <cmath>
thread_local std::string cf_error;
using Tick=std::chrono::steady_clock;
static double seconds(Tick::time_point t){return std::chrono::duration<double>(Tick::now()-t).count();}
static void blas(cublasStatus_t r){if(r!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS error "+std::to_string(r));}
struct Bank {const double* data{};const double* correction{};uint64_t entries{};int n{},tile{},nt{},rank{};
    std::vector<int64_t> offsets,widths;};
struct Context {
    CUmodule module{}; CUfunction scatter_u{},gather_u{},scatter_c{};cublasHandle_t blas_handle{};
    void* host[2]{};CUdeviceptr stage[2]{},left{},right{},row_ids{},col_ids{};CUevent events[2]{};
    bool used[2]{};size_t bytes{};int max_rows{},max_rank{};CfStats stats{};std::vector<Bank> banks;
    ~Context(){for(int i=0;i<2;i++){if(used[i])cuEventSynchronize(events[i]);if(events[i])cuEventDestroy(events[i]);
        if(stage[i])cuMemFree(stage[i]);if(host[i])cuMemFreeHost(host[i]);}
        for(auto p:{left,right,row_ids,col_ids})if(p)cuMemFree(p);
        if(blas_handle)cublasDestroy(blas_handle);if(module)cuModuleUnload(module);}
};
static Context& ctx(void* p){if(!p)throw std::invalid_argument("Null native context");return *static_cast<Context*>(p);}
static Bank& bank(Context& c,int id){if(id<0||id>=int(c.banks.size()))throw std::out_of_range("Unknown bank");return c.banks[id];}
static void indices(const int64_t* p,int n,int bound){if(n<0||(!p&&n))throw std::invalid_argument("Invalid index buffer");
    for(int i=0;i<n;i++)if(p[i]<0||p[i]>=bound)throw std::out_of_range("Index outside designated surface/capacity");}
static void unique_slots(const int64_t* p,int n){std::vector<int64_t> v(p,p+n);std::sort(v.begin(),v.end());
    if(std::adjacent_find(v.begin(),v.end())!=v.end())throw std::invalid_argument("Duplicate region slot");}
static void gather(const Bank& b,const int64_t* rows,int nr,const int64_t* cols,int nc,double* out){
    // Surface IDs repeat across every row. Divide once per requested column,
    // rather than per matrix entry; preserve the caller's exact slot order.
    std::vector<int64_t> tiles(nc),local(nc);
    for(int j=0;j<nc;j++){tiles[j]=cols[j]/b.tile;local[j]=cols[j]%b.tile;}
    for(int i=0;i<nr;i++){int64_t r=rows[i],a=r/b.tile,ri=r%b.tile;
        for(int j=0;j<nc;j++){int64_t d=tiles[j],ci=local[j];
            int64_t k=a<=d?b.offsets[a*b.nt+d]+ri*b.widths[d]+ci:b.offsets[d*b.nt+a]+ci*b.widths[a]+ri;
            out[size_t(i)*nc+j]=b.data[k];}}
}
static void kernel(CUfunction f, uint64_t n, CUstream s, void** args){if(n)cf_check(cuLaunchKernel(f,unsigned((n+255)/256),1,1,256,1,1,0,s,args,nullptr));}
CF_API const char* cf_last_error(){return cf_error.c_str();}
CF_API int cf_create(const char* dir,int rows,int rank,uint64_t bytes,void** output){
    CF_TRY
    if(!output||!dir||rows<1||rank<0||bytes<8*uint64_t(std::max(rows,rank)))throw std::invalid_argument("Invalid bounded staging budget");
    *output=nullptr;cf_check(cuInit(0));CUcontext active{};cf_check(cuCtxGetCurrent(&active));
    if(!active)throw std::runtime_error("Initialize a CUDA context before native backend creation");
    CUdevice device{};cf_check(cuCtxGetDevice(&device));int major{},minor{};
    cf_check(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device));
    cf_check(cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,device));
    if(major<8)throw std::runtime_error("This build requires compute capability >= 8.0");
    auto c=std::make_unique<Context>();c->max_rows=rows;c->max_rank=rank;c->bytes=bytes/8*8;
    auto image=std::filesystem::path(dir)/("region_sm_"+std::to_string(major)+std::to_string(minor)+".cubin");
    if(!std::filesystem::exists(image))image=std::filesystem::path(dir)/"region_compute_86.ptx";
    cf_check(cuModuleLoad(&c->module,image.string().c_str()));
    cf_check(cuModuleGetFunction(&c->scatter_u,c->module,"scatter_factor"));
    cf_check(cuModuleGetFunction(&c->gather_u,c->module,"gather_factor"));
    cf_check(cuModuleGetFunction(&c->scatter_c,c->module,"scatter_response"));
    blas(cublasCreate(&c->blas_handle));
    for(int i=0;i<2;i++){cf_check(cuMemHostAlloc(&c->host[i],c->bytes,0));cf_check(cuMemAlloc(&c->stage[i],c->bytes));
        cf_check(cuEventCreate(&c->events[i],CU_EVENT_DISABLE_TIMING));}
    if(rank){cf_check(cuMemAlloc(&c->left,uint64_t(rows)*rank*8));cf_check(cuMemAlloc(&c->right,uint64_t(std::min(rows,256))*rank*8));}
    cf_check(cuMemAlloc(&c->row_ids,uint64_t(rows)*8));cf_check(cuMemAlloc(&c->col_ids,uint64_t(rows)*8));
    c->stats.host_bytes=2*c->bytes;c->stats.device_bytes=2*c->bytes+uint64_t(rows+std::min(rows,256))*rank*8+uint64_t(rows)*16;
    *output=c.release();CF_CATCH
}
CF_API int cf_destroy(void* p){CF_TRY delete static_cast<Context*>(p);CF_CATCH}
CF_API int cf_register_bank(void* p,const double* data,uint64_t entries,const int64_t* offsets,const int64_t* widths,
    int n,int tile,const double* correction,int rank,int* id){
    CF_TRY
    auto& c=ctx(p);if(!data||!offsets||!widths||!id||n<1||tile<1||rank<0||rank>c.max_rank||(!correction&&rank))throw std::invalid_argument("Invalid bank metadata");
    Bank b;b.data=data;b.entries=entries;b.n=n;b.tile=tile;b.nt=(n+tile-1)/tile;b.correction=correction;b.rank=rank;
    b.offsets.assign(offsets,offsets+size_t(b.nt)*b.nt);b.widths.assign(widths,widths+b.nt);
    for(int a=0;a<b.nt;a++){if(b.widths[a]!=std::min(tile,n-a*tile))throw std::invalid_argument("Invalid packed tile width");
        for(int d=a;d<b.nt;d++){auto offset=b.offsets[a*b.nt+d];
            if(offset<0||uint64_t(offset)+uint64_t(b.widths[a])*std::min(tile,n-d*tile)>entries)throw std::out_of_range("Tile outside mapped bank");}}
    *id=int(c.banks.size());c.banks.push_back(std::move(b));CF_CATCH
}
CF_API int cf_gather(void* p,int id,const int64_t* rows,int nr,const int64_t* cols,int nc,double* out){
    CF_TRY
    auto& b=bank(ctx(p),id);indices(rows,nr,b.n);indices(cols,nc,b.n);if(!out&&nr&&nc)throw std::invalid_argument("Null output");gather(b,rows,nr,cols,nc,out);CF_CATCH
}
CF_API int cf_refill(void* p,int id,uint64_t stream,uint64_t C,uint64_t U,int cap,const int64_t* rows,int nr,
    const int64_t* cols,int nc,const int64_t* ri,const int64_t* cj){
    CF_TRY
    auto start=Tick::now();auto& c=ctx(p);auto& b=bank(c,id);int k=b.rank;
    if(nr>c.max_rows||nc>c.max_rows||cap<1||!C||(k&&!U))throw std::invalid_argument("Region exceeds native allocation");
    indices(rows,nr,b.n);indices(cols,nc,b.n);indices(ri,nr,cap);indices(cj,nc,cap);unique_slots(ri,nr);unique_slots(cj,nc);
    if(!nr||!nc)return 0;
    CUstream s=reinterpret_cast<CUstream>(stream);blas(cublasSetStream(c.blas_handle,reinterpret_cast<cudaStream_t>(s)));
    // Index uploads and factor gather use caller-owned host arrays only until this call returns.
    cf_check(cuStreamSynchronize(s));cf_check(cuMemcpyHtoD(c.row_ids,ri,size_t(nr)*8));cf_check(cuMemcpyHtoD(c.col_ids,cj,size_t(nc)*8));
    auto wait=[&](int slot){auto t=Tick::now();if(c.used[slot])cf_check(cuEventSynchronize(c.events[slot]));c.stats.wait_s+=seconds(t);};
    if(k){int width=int(c.bytes/(8*size_t(k)));
        for(int first=0;first<nc;first+=width){int count=std::min(width,nc-first),slot=int(c.stats.chunks%2);wait(slot);
            auto* h=static_cast<double*>(c.host[slot]);for(int j=0;j<count;j++)std::memcpy(h+size_t(j)*k,b.correction+cols[first+j]*k,size_t(k)*8);
            cf_check(cuMemcpyHtoDAsync(c.stage[slot],h,size_t(count)*k*8,s));
            void* args[]={&c.stage[slot],&U,&c.col_ids,&count,&k,&first};kernel(c.scatter_u,uint64_t(count)*k,s,args);
            cf_check(cuEventRecord(c.events[slot],s));c.used[slot]=true;c.stats.chunks++;c.stats.factor_bytes+=uint64_t(count)*k*8;}
        int first=0;void* args[]={&U,&c.left,&c.row_ids,&nr,&k,&first};kernel(c.gather_u,uint64_t(nr)*k,s,args);}
    int width=std::min(256,int(c.bytes/(8*size_t(nr))));
    for(int first=0;first<nc;first+=width){int count=std::min(width,nc-first),slot=int(c.stats.chunks%2);wait(slot);
        auto* h=static_cast<double*>(c.host[slot]);auto t=Tick::now();gather(b,rows,nr,cols+first,count,h);c.stats.gather_s+=seconds(t);
        cf_check(cuMemcpyHtoDAsync(c.stage[slot],h,size_t(nr)*count*8,s));
        if(k){void* args[]={&U,&c.right,&c.col_ids,&count,&k,&first};kernel(c.gather_u,uint64_t(count)*k,s,args);
            double one=1.;blas(cublasDgemm(c.blas_handle,CUBLAS_OP_T,CUBLAS_OP_N,count,nr,k,&one,
                reinterpret_cast<const double*>(c.right),k,reinterpret_cast<const double*>(c.left),k,&one,reinterpret_cast<double*>(c.stage[slot]),count));}
        for(int transpose=0;transpose<2;transpose++){void* args[]={&c.stage[slot],&C,&c.row_ids,&c.col_ids,&nr,&count,&cap,&first,&transpose};kernel(c.scatter_c,uint64_t(nr)*count,s,args);}
        cf_check(cuEventRecord(c.events[slot],s));c.used[slot]=true;c.stats.chunks++;c.stats.payload_bytes+=uint64_t(nr)*count*8;}
    cf_check(cuStreamSynchronize(s));c.stats.calls++;c.stats.total_s+=seconds(start);CF_CATCH
}
CF_API int cf_stats(void* p,CfStats* out,int reset){CF_TRY auto& c=ctx(p);if(!out)throw std::invalid_argument("Null statistics");*out=c.stats;
    if(reset){auto h=c.stats.host_bytes,d=c.stats.device_bytes;c.stats={};c.stats.host_bytes=h;c.stats.device_bytes=d;}CF_CATCH}

struct RegionBody {
    CfRegionBody d{};std::vector<int> faces,required,active,lookup;
    std::vector<int64_t> slots,age;
};
struct Region {
    Context* refill{};CfRegionConfig cfg{};std::vector<RegionBody> bodies;std::vector<CfRegionPair> pairs;
    std::vector<int> mask;CUdeviceptr device_mask{},ids{},dest{};
    CUfunction mark_pairs{},mark_load{},mark_ready{},modal_rows{};
    CfRegionStats stats{};
    ~Region(){for(auto p:{device_mask,ids,dest})if(p)cuMemFree(p);}
};
static Region& region(void* p){if(!p)throw std::invalid_argument("Null region runtime");return *static_cast<Region*>(p);}
CF_API int cf_region_create(void* refill,const CfRegionConfig* cfg,void** output){
    CF_TRY
    if(!cfg||!output||!cfg->bodies||cfg->body_count<1||cfg->node_count<1||cfg->rings<1||cfg->rings>6||
       cfg->history_fraction<0||cfg->history_fraction>1)throw std::invalid_argument("Invalid native regional configuration");
    *output=nullptr;auto r=std::make_unique<Region>();r->refill=&ctx(refill);r->cfg=*cfg;
    r->pairs.assign(cfg->pairs,cfg->pairs+cfg->pair_count);r->mask.resize(cfg->node_count);
    for(int i=0;i<cfg->body_count;i++){
        const auto& d=cfg->bodies[i];
        if(d.n<1||d.n%3||d.capacity<1||d.capacity>r->refill->max_rows||!d.faces||!d.slots||!d.age||!d.required)
            throw std::invalid_argument("Invalid regional body");
        bank(*r->refill,d.bank);RegionBody b;b.d=d;
        b.faces.assign(d.faces,d.faces+size_t(d.face_count)*3);
        indices(d.slots,0,d.n); // Empty slots use -1; validate separately.
        for(int j=0;j<d.capacity;j++)if(d.slots[j]<-1||d.slots[j]>=d.n)throw std::invalid_argument("Invalid resident slot");
        b.slots.assign(d.slots,d.slots+d.capacity);b.age.assign(d.age,d.age+d.n);
        b.required.assign(d.required,d.required+d.n);b.active=b.required;b.lookup.resize(d.n,-1);
        for(auto v:b.faces)if(v<0||v>=d.n/3)throw std::out_of_range("Face outside designated body");
        for(int j=0;j<d.capacity;j++)if(b.slots[j]>=0){if(b.lookup[b.slots[j]]!=-1)throw std::invalid_argument("Duplicate resident column");b.lookup[b.slots[j]]=j;}
        r->stats.host_bytes+=uint64_t(b.faces.size()+b.required.size()+b.active.size()+b.lookup.size())*4+
            uint64_t(b.slots.size()+b.age.size())*8;
        r->bodies.push_back(std::move(b));
    }
    cf_check(cuMemAlloc(&r->device_mask,size_t(cfg->node_count)*4));
    cf_check(cuMemAlloc(&r->ids,size_t(r->refill->max_rows)*8));cf_check(cuMemAlloc(&r->dest,size_t(r->refill->max_rows)*8));
    r->stats.device_bytes=uint64_t(cfg->node_count)*4+uint64_t(r->refill->max_rows)*16;
    r->stats.host_bytes+=uint64_t(r->mask.size())*4;
    cf_check(cuModuleGetFunction(&r->mark_pairs,r->refill->module,"region_mark_pairs"));
    cf_check(cuModuleGetFunction(&r->mark_load,r->refill->module,"region_mark_load"));
    cf_check(cuModuleGetFunction(&r->mark_ready,r->refill->module,"region_mark_ready"));
    cf_check(cuModuleGetFunction(&r->modal_rows,r->refill->module,"region_modal_rows"));
    *output=r.release();CF_CATCH
}
CF_API int cf_region_destroy(void* p){CF_TRY delete static_cast<Region*>(p);CF_CATCH}
CF_API int cf_region_clear_required(void* p){CF_TRY for(auto& b:region(p).bodies)std::fill(b.required.begin(),b.required.end(),0);CF_CATCH}
CF_API int cf_region_export(void* p,int i,int64_t* slots,int64_t* age,int* required,int* active,int64_t* epoch){
    CF_TRY auto& r=region(p);if(i<0||i>=int(r.bodies.size())||!slots||!age||!required||!active||!epoch)throw std::invalid_argument("Invalid region export");
    auto& b=r.bodies[i];std::copy(b.slots.begin(),b.slots.end(),slots);std::copy(b.age.begin(),b.age.end(),age);
    std::copy(b.required.begin(),b.required.end(),required);std::copy(b.active.begin(),b.active.end(),active);*epoch=r.cfg.epoch;CF_CATCH
}
CF_API int cf_region_stats(void* p,CfRegionStats* out){CF_TRY if(!out)throw std::invalid_argument("Null region stats");*out=region(p).stats;CF_CATCH}
CF_API int cf_region_envelope(void* p,uint64_t stream){
    CF_TRY auto& r=region(p);cf_check(cuStreamSynchronize(reinterpret_cast<CUstream>(stream)));
    double envelope=0.,maximum=0.;cf_check(cuMemcpyDtoH(&envelope,r.cfg.envelope,8));
    for(auto& b:r.bodies){double bound=0.;cf_check(cuMemcpyDtoH(&bound,b.d.bound,8));
        if(!std::isfinite(bound)||bound<0)throw std::runtime_error("Invalid residual motion certificate");maximum=std::max(maximum,bound);}
    envelope=std::nextafter(1.25*std::max(envelope,maximum),INFINITY);
    if(envelope>.01)throw std::runtime_error("Residual envelope exceeded 10 mm");
    double radius=r.cfg.base_query_radius+2*envelope;int force=1;
    cf_check(cuMemcpyHtoD(r.cfg.envelope,&envelope,8));cf_check(cuMemcpyHtoD(r.cfg.query_radius,&radius,8));
    cf_check(cuMemcpyHtoD(r.cfg.force_query,&force,4));CF_CATCH
}
CF_API int cf_region_update(void* p,uint64_t stream,int* needs_capacity){
    CF_TRY
    auto start=Tick::now();auto& r=region(p);auto s=reinterpret_cast<CUstream>(stream);if(!needs_capacity)throw std::invalid_argument("Null capacity result");
    *needs_capacity=0;cf_check(cuMemsetD32Async(r.device_mask,0,r.mask.size(),s));
    for(auto pair:r.pairs){void* args[]={&pair.owners,&pair.mids,&pair.slave_faces,&pair.master_faces,&r.device_mask,&pair.count,&pair.slave_offset,&pair.master_offset};kernel(r.mark_pairs,pair.count,s,args);}
    if(r.cfg.csr_capacity){void* args[]={&r.cfg.ptr,&r.cfg.col,&r.cfg.val,&r.device_mask,&r.cfg.csr_capacity,&r.cfg.csr_rows};
        cf_check(cuLaunchKernel(r.mark_load,256,1,1,256,1,1,0,s,args,nullptr));}
    cf_check(cuStreamSynchronize(s));cf_check(cuMemcpyDtoH(r.mask.data(),r.device_mask,r.mask.size()*4));
    // Expand each normal halo from the unmodified round's node mask, exactly
    // like the vectorized reference. Retain all requests within a failed step.
    std::vector<std::vector<int>> requested;
    for(auto& b:r.bodies){int nodes=b.d.n/3;std::vector<int> seen(r.mask.begin()+b.d.node_offset,r.mask.begin()+b.d.node_offset+nodes);
        for(int ring=0;ring<r.cfg.rings;ring++){auto expanded=seen;
            for(int f=0;f<b.d.face_count;f++){int a=b.faces[3*f],c=b.faces[3*f+1],d=b.faces[3*f+2];
                if(seen[a]||seen[c]||seen[d])expanded[a]=expanded[c]=expanded[d]=1;}
            seen.swap(expanded);}
        auto active=b.required;int count=0;
        for(int j=0;j<b.d.n;j++){active[j]=active[j]||seen[j/3];count+=active[j]!=0;}
        if(count>b.d.capacity)*needs_capacity=1;requested.push_back(std::move(active));
    }
    if(*needs_capacity){r.stats.update_s+=seconds(start);return 0;}
    ++r.cfg.epoch;
    for(size_t bi=0;bi<r.bodies.size();bi++){
        auto& b=r.bodies[bi];auto& d=b.d;b.active=requested[bi];b.required=b.active;
        std::vector<int64_t> missing,dest,occupied,all_ids;int active_count=0;
        for(int j=0;j<d.n;j++)if(b.active[j]){active_count++;b.age[j]=r.cfg.epoch;if(b.lookup[j]<0)missing.push_back(j);}
        for(int j=0;j<d.capacity;j++)if(b.slots[j]<0)dest.push_back(j);
        for(int j=0;j<d.capacity;j++)if(b.slots[j]>=0&&!b.active[b.slots[j]])dest.push_back(j);
        if(dest.size()<missing.size())throw std::runtime_error("No evictable native region slots");dest.resize(missing.size());
        for(size_t j=0;j<missing.size();j++){auto previous=b.slots[dest[j]];if(previous>=0){b.lookup[previous]=-1;r.stats.evicted_columns++;}b.slots[dest[j]]=missing[j];}
        for(int j=0;j<d.capacity;j++)if(b.slots[j]>=0){occupied.push_back(j);all_ids.push_back(b.slots[j]);}
        if(!missing.empty()){
            if(cf_refill(r.refill,d.bank,stream,d.C,d.U,d.capacity,all_ids.data(),int(all_ids.size()),missing.data(),int(missing.size()),occupied.data(),dest.data()))
                throw std::runtime_error(cf_error);
            cf_check(cuMemcpyHtoD(r.ids,missing.data(),missing.size()*8));cf_check(cuMemcpyHtoD(r.dest,dest.data(),dest.size()*8));
            int n=int(missing.size());if(d.modes){void* args[]={&d.P,&d.inverse,&d.D,&r.ids,&r.dest,&n,&d.modes};
                cf_check(cuLaunchKernel(r.modal_rows,(d.modes+127)/128,n,1,128,1,1,0,s,args,nullptr));}
            for(size_t j=0;j<missing.size();j++)b.lookup[missing[j]]=int(dest[j]);
            r.stats.new_columns+=missing.size();
        }
        // Stable age ordering is resolved by increasing surface ID, matching
        // np.argsort(kind='stable') over the slot-ordered history below.
        std::vector<int64_t> history;for(auto id:b.slots)if(id>=0&&!b.active[id])history.push_back(id);
        int limit=int(r.cfg.history_fraction*active_count);
        if(int(history.size())>limit){std::stable_sort(history.begin(),history.end(),[&](auto a,auto c){return b.age[a]<b.age[c];});
            for(size_t j=0;j<history.size()-limit;j++){auto id=history[j];b.slots[b.lookup[id]]=-1;b.lookup[id]=-1;}}
        // Publish only after both triangular entries and the dynamic mapping
        // have completed. Subsequent geometry never sees a partial refill.
        cf_check(cuStreamSynchronize(s));cf_check(cuMemcpyHtoD(d.lookup,b.lookup.data(),size_t(d.n)*4));
        void* args[]={&d.halo_ptr,&d.halo_nodes,&d.lookup,&d.ready,&d.face_count};kernel(r.mark_ready,d.face_count,s,args);
    }
    cf_check(cuMemsetD32Async(r.cfg.oldcount,0,1,s));cf_check(cuStreamSynchronize(s));
    r.stats.updates++;r.stats.update_s+=seconds(start);CF_CATCH
}
