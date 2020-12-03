#include <CL/sycl.hpp>
#include "dppl_sycl_types.h"

extern int c_columnwise_total(
    DPPLSyclQueueRef q, size_t n, size_t m, double *mat, double *ct);
extern int c_columnwise_total_no_mkl(
    DPPLSyclQueueRef q, size_t n, size_t m, double *mat, double *ct);
