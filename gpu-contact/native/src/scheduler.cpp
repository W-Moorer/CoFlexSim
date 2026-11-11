#include "coflex_native.h"
#include "internal.h"
#include <limits>
#include <chrono>
#include <algorithm>
CF_API int cf_graph_batch(uint64_t exec, uint64_t stream, uint64_t device_run,
    uint64_t clock, double next_time, int steps, uint64_t history,
    int width, double* host_history, int* state) {
    CF_TRY
    if(!exec||!device_run||!clock||!history||!host_history||!state||steps<1||width<1)
        throw std::invalid_argument("Invalid native batch buffers");
    CUstream s=reinterpret_cast<CUstream>(stream);
    int run[3]={steps,0,0};
    // These small host values are synchronous copies: stack lifetime is explicit.
    cf_check(cuStreamSynchronize(s));
    cf_check(cuMemcpyHtoD(device_run,run,sizeof(run)));
    cf_check(cuMemcpyHtoD(clock,&next_time,sizeof(next_time)));
    cf_check(cuGraphLaunch(reinterpret_cast<CUgraphExec>(exec),s));
    cf_check(cuStreamSynchronize(s));
    cf_check(cuMemcpyDtoH(state,device_run,sizeof(run)));
    if(state[1]<0||state[1]>steps)throw std::runtime_error("Invalid accepted step count");
    if(state[1])cf_check(cuMemcpyDtoH(host_history,history,size_t(state[1])*width*sizeof(double)));
    CF_CATCH
}

CF_API int cf_advance(CfLaunch* l,int steps,double* rows,int* accepted,
                     CfRepair repair,void* user,CfAdvanceStats* stats){
    CF_TRY
    if(!l||!rows||!accepted||!stats||steps<1||l->width<1||l->batch_capacity<1||l->dt<=0)
        throw std::invalid_argument("Invalid native trajectory request");
    *accepted=0;*stats={};int retry=0;
    using Clock=std::chrono::steady_clock;auto start=Clock::now();
    while(*accepted<steps){
        int count=std::min(l->batch_capacity,steps-*accepted),state[3]{};
        double* target=rows+size_t(*accepted)*l->width;
        if(cf_graph_batch(l->exec,l->stream,l->run,l->clock,l->time+l->dt,count,l->history,l->width,target,state))
            throw std::runtime_error(cf_error);
        stats->launches++;*accepted+=state[1];
        if(state[1]){
            l->time=target[size_t(state[1]-1)*l->width];retry=0;
            if(l->region&&cf_region_clear_required(l->region))throw std::runtime_error(cf_error);
        }
        if(!state[2]){
            if(state[1]!=count)throw std::runtime_error("Native batch stopped without a reported error");
            continue;
        }
        if(++retry>128)throw std::runtime_error("Native retry budget exceeded");
        if(state[2]==16384&&l->region){
            if(cf_region_envelope(l->region,l->stream))throw std::runtime_error(cf_error);
            stats->envelope_retries++;continue;
        }
        if(state[2]==8192&&l->region){
            int needs=0;
            if(cf_region_update(l->region,l->stream,&needs))throw std::runtime_error(cf_error);
            stats->region_retries++;
            if(!needs)continue;
        }
        // Arena replacement currently uses the explicitly registered migration
        // adapter. No callback runs within ordinary equilibrium/contact loops.
        if(!repair)throw std::runtime_error("Native arena/capture repair required: "+std::to_string(state[2]));
        auto tick=Clock::now();
        if(repair(user,l,state[2],*accepted,rows))throw std::runtime_error("Native repair adapter failed");
        stats->host_repairs++;stats->repair_s+=std::chrono::duration<double>(Clock::now()-tick).count();
    }
    stats->solve_s=std::chrono::duration<double>(Clock::now()-start).count();CF_CATCH
}
