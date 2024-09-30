#pragma once

#include <utility>
#include <vector>

#include "dpctl4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <sycl/sycl.hpp>

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

std::pair<sycl::event, sycl::event>
py_as_c_contig(const dpctl::tensor::usm_ndarray &,
               const dpctl::tensor::usm_ndarray &,
               sycl::queue &,
               const std::vector<sycl::event> &);

std::pair<sycl::event, sycl::event>
py_as_f_contig(const dpctl::tensor::usm_ndarray &,
               const dpctl::tensor::usm_ndarray &,
               sycl::queue &,
               const std::vector<sycl::event> &);

void init_copy_as_contig_dispatch_vectors(void);

} // end of namespace py_internal
} // end of namespace tensor
} // end of namespace dpctl
