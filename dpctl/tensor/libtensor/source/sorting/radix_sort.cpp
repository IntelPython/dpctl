#include <cstdint>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpctl4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utils/memory_overlap.hpp"
#include "utils/offset_utils.hpp"
#include "utils/output_validation.hpp"
#include "utils/sycl_alloc_utils.hpp"
#include "utils/type_dispatch.hpp"

#include "kernels/sorting/radix_sort.hpp"
#include "radix_sort_support.hpp"

#include "py_sort_common.hpp"
#include "radix_sort.hpp"

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

namespace td_ns = dpctl::tensor::type_dispatch;
namespace impl_ns = dpctl::tensor::kernels::radix_sort_details;

using dpctl::tensor::kernels::radix_sort_contig_fn_ptr_t;
static radix_sort_contig_fn_ptr_t
    ascending_radix_sort_contig_dispatch_vector[td_ns::num_types];
static radix_sort_contig_fn_ptr_t
    descending_radix_sort_contig_dispatch_vector[td_ns::num_types];

template <typename fnT, typename argTy> struct AscendingRadixSortContigFactory
{
    fnT get()
    {
        if constexpr (RadixSortSupportVector<argTy>::is_defined) {
            using dpctl::tensor::kernels::radix_sort_axis1_contig_impl;
            return radix_sort_axis1_contig_impl<argTy, /*ascending*/ true>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename argTy> struct DescendingRadixSortContigFactory
{
    fnT get()
    {
        if constexpr (RadixSortSupportVector<argTy>::is_defined) {
            using dpctl::tensor::kernels::radix_sort_axis1_contig_impl;
            return radix_sort_axis1_contig_impl<argTy, /*ascending*/ false>;
        }
        else {
            return nullptr;
        }
    }
};

void init_radix_sort_dispatch_vectors(void)
{
    using dpctl::tensor::kernels::radix_sort_contig_fn_ptr_t;

    td_ns::DispatchVectorBuilder<radix_sort_contig_fn_ptr_t,
                                 AscendingRadixSortContigFactory,
                                 td_ns::num_types>
        dtv1;
    dtv1.populate_dispatch_vector(ascending_radix_sort_contig_dispatch_vector);

    td_ns::DispatchVectorBuilder<radix_sort_contig_fn_ptr_t,
                                 DescendingRadixSortContigFactory,
                                 td_ns::num_types>
        dtv2;
    dtv2.populate_dispatch_vector(descending_radix_sort_contig_dispatch_vector);
}

void init_radix_sort_functions(py::module_ m)
{
    dpctl::tensor::py_internal::init_radix_sort_dispatch_vectors();

    auto py_radix_sort_ascending = [](const dpctl::tensor::usm_ndarray &src,
                                      const int trailing_dims_to_sort,
                                      const dpctl::tensor::usm_ndarray &dst,
                                      sycl::queue &exec_q,
                                      const std::vector<sycl::event> &depends)
        -> std::pair<sycl::event, sycl::event> {
        return dpctl::tensor::py_internal::py_sort(
            src, trailing_dims_to_sort, dst, exec_q, depends,
            dpctl::tensor::py_internal::
                ascending_radix_sort_contig_dispatch_vector);
    };
    m.def("_radix_sort_ascending", py_radix_sort_ascending, py::arg("src"),
          py::arg("trailing_dims_to_sort"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    auto py_radix_sort_descending = [](const dpctl::tensor::usm_ndarray &src,
                                       const int trailing_dims_to_sort,
                                       const dpctl::tensor::usm_ndarray &dst,
                                       sycl::queue &exec_q,
                                       const std::vector<sycl::event> &depends)
        -> std::pair<sycl::event, sycl::event> {
        return dpctl::tensor::py_internal::py_sort(
            src, trailing_dims_to_sort, dst, exec_q, depends,
            dpctl::tensor::py_internal::
                descending_radix_sort_contig_dispatch_vector);
    };
    m.def("_radix_sort_descending", py_radix_sort_descending, py::arg("src"),
          py::arg("trailing_dims_to_sort"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    return;
}

} // namespace py_internal
} // end of namespace tensor
} // end of namespace dpctl
