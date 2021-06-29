#include "dpcpp_kernels.hpp"
#include <CL/sycl.hpp>
#include <cstddef>

template sycl::kernel
dpcpp_kernels::get_fill_kernel<int>(sycl::queue &, size_t, int *, int);

template sycl::kernel
dpcpp_kernels::get_fill_kernel<unsigned int>(sycl::queue &,
                                             size_t,
                                             unsigned int *,
                                             unsigned int);

template sycl::kernel
dpcpp_kernels::get_fill_kernel<double>(sycl::queue &, size_t, double *, double);

template sycl::kernel
dpcpp_kernels::get_fill_kernel<float>(sycl::queue &, size_t, float *, float);

template sycl::kernel
dpcpp_kernels::get_range_kernel<int>(sycl::queue &, size_t, int *);

template sycl::kernel
dpcpp_kernels::get_range_kernel<unsigned int>(sycl::queue &,
                                              size_t,
                                              unsigned int *);

template sycl::kernel
dpcpp_kernels::get_range_kernel<float>(sycl::queue &, size_t, float *);

template sycl::kernel
dpcpp_kernels::get_range_kernel<double>(sycl::queue &, size_t, double *);

template sycl::kernel dpcpp_kernels::get_mad_kernel<int, int>(sycl::queue &,
                                                              size_t,
                                                              int *,
                                                              int *,
                                                              int *,
                                                              int);

template sycl::kernel
dpcpp_kernels::get_mad_kernel<unsigned int, unsigned int>(sycl::queue &,
                                                          size_t,
                                                          unsigned int *,
                                                          unsigned int *,
                                                          unsigned int *,
                                                          unsigned int);

template sycl::kernel dpcpp_kernels::get_local_sort_kernel<int>(sycl::queue &,
                                                                size_t,
                                                                size_t,
                                                                int *,
                                                                size_t);

template sycl::kernel
dpcpp_kernels::get_local_count_exceedance_kernel<int>(sycl::queue &,
                                                      size_t,
                                                      size_t,
                                                      int *,
                                                      size_t,
                                                      int,
                                                      int *);

template sycl::kernel
dpcpp_kernels::get_local_count_exceedance_kernel<unsigned int>(sycl::queue &,
                                                               size_t,
                                                               size_t,
                                                               unsigned int *,
                                                               size_t,
                                                               unsigned int,
                                                               int *);

template sycl::kernel
dpcpp_kernels::get_local_count_exceedance_kernel<float>(sycl::queue &,
                                                        size_t,
                                                        size_t,
                                                        float *,
                                                        size_t,
                                                        float,
                                                        int *);

template sycl::kernel
dpcpp_kernels::get_local_count_exceedance_kernel<double>(sycl::queue &,
                                                         size_t,
                                                         size_t,
                                                         double *,
                                                         size_t,
                                                         double,
                                                         int *);
