#include "dpcpp_kernels.hpp"
#include <CL/sycl.hpp>
#include <cstddef>

template sycl::kernel
dpcpp_kernels::get_fill_kernel<int>(sycl::queue &, size_t, int *, int);

template sycl::kernel
dpcpp_kernels::get_range_kernel<int>(sycl::queue &, size_t, int *);

template sycl::kernel dpcpp_kernels::get_mad_kernel<int, int>(sycl::queue &,
                                                              size_t,
                                                              int *,
                                                              int *,
                                                              int *,
                                                              int);
