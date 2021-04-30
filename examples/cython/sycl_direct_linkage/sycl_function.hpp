#include <CL/sycl.hpp>

int c_columnwise_total(cl::sycl::queue &,
                       size_t n,
                       size_t m,
                       double *mat,
                       double *ct);
