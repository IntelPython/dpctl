#include <CL/sycl.hpp>
#include "dppl_sycl_types.h"

template<typename T>
extern void cpp_blackscholes(DPPLSyclQueueRef q, size_t n_opts, T* params, T* callput); 

template<typename T>
extern void cpp_populate_params(DPPLSyclQueueRef q, size_t n_opts, T* params,
				T pl, T ph, T sl, T sh, T tl, T th, T rl, T rh, T vl, T vh,
				int seed); 
