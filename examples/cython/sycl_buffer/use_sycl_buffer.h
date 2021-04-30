#include "dpctl_sycl_types.h"
#include <CL/sycl.hpp>

extern int c_columnwise_total(DPCTLSyclQueueRef q,
                              size_t n,
                              size_t m,
                              double *mat,
                              double *ct);
extern int c_columnwise_total_no_mkl(DPCTLSyclQueueRef q,
                                     size_t n,
                                     size_t m,
                                     double *mat,
                                     double *ct);
