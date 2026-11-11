#pragma once
#include <cuda.h>
#include <stdexcept>
#include <string>
extern thread_local std::string cf_error;
inline void cf_check(CUresult r) {
    if(r!=CUDA_SUCCESS) {const char* s=nullptr;cuGetErrorString(r,&s);
        throw std::runtime_error(s?s:"CUDA driver error");}
}
#define CF_TRY try { cf_error.clear();
#define CF_CATCH return 0; } catch(const std::exception& e) {cf_error=e.what();return -1;} catch(...) {cf_error="Unknown native error";return -2;}
