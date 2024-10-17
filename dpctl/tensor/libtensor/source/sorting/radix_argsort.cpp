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

#include "py_argsort_common.hpp"
#include "radix_argsort.hpp"

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
    ascending_radix_argsort_contig_dispatch_table[td_ns::num_types]
                                                 [td_ns::num_types];
static radix_sort_contig_fn_ptr_t
    descending_radix_argsort_contig_dispatch_table[td_ns::num_types]
                                                  [td_ns::num_types];

template <typename fnT, typename argTy, typename IndexTy>
struct AscendingRadixArgSortContigFactory
{
    fnT get()
    {
        if constexpr (RadixSortSupportVector<argTy>::is_defined &&
                      (std::is_same_v<IndexTy, std::int64_t> ||
                       std::is_same_v<IndexTy, std::int32_t>))
        {
            using dpctl::tensor::kernels::radix_argsort_axis1_contig_alt_impl;
            return radix_argsort_axis1_contig_alt_impl<argTy, IndexTy,
                                                       /* ascending= */ true>;
        }
        else {
            return nullptr;
        }
    }
};

template <typename fnT, typename argTy, typename IndexTy>
struct DescendingRadixArgSortContigFactory
{
    fnT get()
    {
        if constexpr (RadixSortSupportVector<argTy>::is_defined &&
                      (std::is_same_v<IndexTy, std::int64_t> ||
                       std::is_same_v<IndexTy, std::int32_t>))
        {
            using dpctl::tensor::kernels::radix_argsort_axis1_contig_alt_impl;
            return radix_argsort_axis1_contig_alt_impl<argTy, IndexTy,
                                                       /* ascending= */ false>;
        }
        else {
            return nullptr;
        }
    }
};

void init_radix_argsort_dispatch_tables(void)
{
    using dpctl::tensor::kernels::radix_sort_contig_fn_ptr_t;

    td_ns::DispatchTableBuilder<radix_sort_contig_fn_ptr_t,
                                AscendingRadixArgSortContigFactory,
                                td_ns::num_types>
        dtb1;
    dtb1.populate_dispatch_table(ascending_radix_argsort_contig_dispatch_table);

    td_ns::DispatchTableBuilder<radix_sort_contig_fn_ptr_t,
                                DescendingRadixArgSortContigFactory,
                                td_ns::num_types>
        dtb2;
    dtb2.populate_dispatch_table(
        descending_radix_argsort_contig_dispatch_table);
}

void init_radix_argsort_functions(py::module_ m)
{
    dpctl::tensor::py_internal::init_radix_argsort_dispatch_tables();

    auto py_radix_argsort_ascending =
        [](const dpctl::tensor::usm_ndarray &src,
           const int trailing_dims_to_sort,
           const dpctl::tensor::usm_ndarray &dst, sycl::queue &exec_q,
           const std::vector<sycl::event> &depends)
        -> std::pair<sycl::event, sycl::event> {
        return dpctl::tensor::py_internal::py_argsort(
            src, trailing_dims_to_sort, dst, exec_q, depends,
            dpctl::tensor::py_internal::
                ascending_radix_argsort_contig_dispatch_table);
    };
    m.def("_radix_argsort_ascending", py_radix_argsort_ascending,
          py::arg("src"), py::arg("trailing_dims_to_sort"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    auto py_radix_argsort_descending =
        [](const dpctl::tensor::usm_ndarray &src,
           const int trailing_dims_to_sort,
           const dpctl::tensor::usm_ndarray &dst, sycl::queue &exec_q,
           const std::vector<sycl::event> &depends)
        -> std::pair<sycl::event, sycl::event> {
        return dpctl::tensor::py_internal::py_argsort(
            src, trailing_dims_to_sort, dst, exec_q, depends,
            dpctl::tensor::py_internal::
                descending_radix_argsort_contig_dispatch_table);
    };
    m.def("_radix_argsort_descending", py_radix_argsort_descending,
          py::arg("src"), py::arg("trailing_dims_to_sort"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    return;
}

} // namespace py_internal
} // end of namespace tensor
} // end of namespace dpctl
