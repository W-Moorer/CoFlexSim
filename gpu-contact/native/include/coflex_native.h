#pragma once
#include <stdint.h>
#if defined(_WIN32)
# if defined(COFLEX_EXPORTS)
#  define CF_API extern "C" __declspec(dllexport)
# else
#  define CF_API extern "C" __declspec(dllimport)
# endif
#else
# define CF_API extern "C" __attribute__((visibility("default")))
#endif
// ABI version 1. All arrays are FP64/int64; explicit device pointers are uint64.
// No exception crosses the C ABI. A returned error leaves the caller's trial
// unaccepted. The enclosing solver must not commit on a nonzero return.
struct CfStats {
    uint64_t calls, chunks, payload_bytes, factor_bytes, host_bytes, device_bytes;
    double gather_s, wait_s, total_s;
};
CF_API const char* cf_last_error();
CF_API int cf_create(const char* image_directory, int max_rows, int max_rank,
                     uint64_t slot_bytes, void** output);
CF_API int cf_destroy(void* handle);
// Register a borrowed, read-only mapping. Owner must retain it until destroy.
// The complete bank is not copied into RAM or VRAM.
CF_API int cf_register_bank(void* handle, const double* data, uint64_t entries,
    const int64_t* offsets, const int64_t* widths, int n, int tile,
    const double* correction, int rank, int* bank_id);
CF_API int cf_gather(void* handle, int bank_id, const int64_t* rows, int nr,
                    const int64_t* cols, int nc, double* output);
// C is column-major cap x cap. U is row-major cap x rank. New regional entries
// are R_base + U U^T; cross-region terms and symmetry are retained.
CF_API int cf_refill(void* handle, int bank_id, uint64_t stream,
    uint64_t C, uint64_t U, int capacity, const int64_t* rows, int nr,
    const int64_t* cols, int nc, const int64_t* row_slots, const int64_t* col_slots);
CF_API int cf_stats(void* handle, CfStats* stats, int reset_counters);
// C++ owns submission/status/history copies for a single conditional batch.
// Cache/capacity policy still lives in the migration adapter until ported.
CF_API int cf_graph_batch(uint64_t exec, uint64_t stream, uint64_t device_run,
    uint64_t clock, double next_time, int steps, uint64_t device_history,
    int row_width, double* host_history, int* host_state);

// Runtime descriptors borrow immutable model arrays and stable GPU arenas.
// Host metadata is copied; device storage remains owned by the model adapter.
struct CfRegionBody {
    int n, capacity, modes, bank, face_count, node_offset;
    uint64_t C,U,P,inverse,D,lookup,ready,halo_ptr,halo_nodes,bound;
    const int* faces;
    const int64_t* slots;
    const int64_t* age;
    const int* required;
};
struct CfRegionPair {uint64_t owners,mids,slave_faces,master_faces;int count,slave_offset,master_offset;};
struct CfRegionConfig {
    const CfRegionBody* bodies;int body_count;
    const CfRegionPair* pairs;int pair_count;
    uint64_t ptr,col,val,oldcount;
    int csr_rows,csr_capacity,node_count,rings;
    double history_fraction;
    int64_t epoch;
    uint64_t envelope,query_radius,force_query;
    double base_query_radius;
};
struct CfRegionStats {uint64_t updates,new_columns,evicted_columns,device_bytes,host_bytes;double update_s;};
CF_API int cf_region_create(void* refill,const CfRegionConfig* config,void** output);
CF_API int cf_region_destroy(void* region);
// 0 succeeds; 1 requests a larger declared arena (no partial mutation).
CF_API int cf_region_update(void* region,uint64_t stream,int* needs_capacity);
CF_API int cf_region_clear_required(void* region);
CF_API int cf_region_export(void* region,int body,int64_t* slots,int64_t* age,int* required,int* active,int64_t* epoch);
CF_API int cf_region_stats(void* region,CfRegionStats* result);
CF_API int cf_region_envelope(void* region,uint64_t stream);

struct CfLaunch {
    uint64_t exec,stream,run,clock,history;
    int width,batch_capacity;
    double time,dt;
    void* region;
};
// A capacity/graph repair callback is an explicit migration boundary. It is
// never called for ordinary steps or an in-capacity regional refill.
typedef int (*CfRepair)(void* user,CfLaunch* launch,int error,int accepted,const double* rows);
struct CfAdvanceStats {uint64_t launches,region_retries,envelope_retries,host_repairs;double solve_s,repair_s;};
CF_API int cf_advance(CfLaunch* launch,int steps,double* rows,int* accepted,
                     CfRepair repair,void* user,CfAdvanceStats* stats);
