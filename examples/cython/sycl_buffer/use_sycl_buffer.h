#include <CL/sycl.hpp>
#include "dppl_sycl_types.h"

extern int c_columnwise_total(
    DPPLSyclQueueRef q, size_t n, size_t m, double *mat, double *ct);
