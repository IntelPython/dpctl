//===-- ------------ Implementation of _tensor_impl module  ----*-C++-*-/===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===--------------------------------------------------------------------===//
///
/// \file
/// This file defines functions of dpctl.tensor._tensor_impl extensions
//===--------------------------------------------------------------------===//

#include "dpctl4pybind11.hpp"
#include <complex>
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

#include "kernels/constructors.hpp"
#include "utils/output_validation.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

#include "full_ctor.hpp"
#include "unboxing_helper.hpp"

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

using dpctl::utils::keep_args_alive;

typedef sycl::event (*full_contig_fn_ptr_t)(sycl::queue &,
                                            size_t,
                                            const py::object &,
                                            char *,
                                            const std::vector<sycl::event> &);

/*!
 * @brief Function to submit kernel to fill given contiguous memory allocation
 * with specified value.
 *
 * @param exec_q  Sycl queue to which kernel is submitted for execution.
 * @param nelems  Length of the sequence
 * @param py_value Python object representing the value to fill the array with.
 * Must be convertible to `dstTy`.
 * @param dst_p Kernel accessible USM pointer to the start of array to be
 * populated.
 * @param depends  List of events to wait for before starting computations, if
 * any.
 *
 * @return Event to wait on to ensure that computation completes.
 * @defgroup CtorKernels
 */
template <typename dstTy>
sycl::event full_contig_impl(sycl::queue &exec_q,
                             size_t nelems,
                             const py::object &py_value,
                             char *dst_p,
                             const std::vector<sycl::event> &depends)
{
    dstTy fill_v;

    PythonObjectUnboxer<dstTy> unboxer{};
    try {
        fill_v = unboxer(py_value);
    } catch (const py::error_already_set &e) {
        throw;
    }

    using dpctl::tensor::kernels::constructors::full_contig_impl;

    sycl::event fill_ev =
        full_contig_impl<dstTy>(exec_q, nelems, fill_v, dst_p, depends);

    return fill_ev;
}

template <typename fnT, typename Ty> struct FullContigFactory
{
    fnT get()
    {
        fnT f = full_contig_impl<Ty>;
        return f;
    }
};

static full_contig_fn_ptr_t full_contig_dispatch_vector[td_ns::num_types];

std::pair<sycl::event, sycl::event>
usm_ndarray_full(const py::object &py_value,
                 const dpctl::tensor::usm_ndarray &dst,
                 sycl::queue &exec_q,
                 const std::vector<sycl::event> &depends)
{
    // start, end should be coercible into data type of dst

    py::ssize_t dst_nelems = dst.get_size();

    if (dst_nelems == 0) {
        // nothing to do
        return std::make_pair(sycl::event(), sycl::event());
    }

    if (!dpctl::utils::queues_are_compatible(exec_q, {dst})) {
        throw py::value_error(
            "Execution queue is not compatible with the allocation queue");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    auto array_types = td_ns::usm_ndarray_types();
    int dst_typenum = dst.get_typenum();
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    char *dst_data = dst.get_data();
    sycl::event full_event;

    if (dst_nelems == 1 || dst.is_c_contiguous() || dst.is_f_contiguous()) {
        auto fn = full_contig_dispatch_vector[dst_typeid];

        sycl::event full_contig_event =
            fn(exec_q, static_cast<size_t>(dst_nelems), py_value, dst_data,
               depends);

        return std::make_pair(
            keep_args_alive(exec_q, {dst}, {full_contig_event}),
            full_contig_event);
    }
    else {
        throw std::runtime_error(
            "Only population of contiguous usm_ndarray objects is supported.");
    }
}

void init_full_ctor_dispatch_vectors(void)
{
    using namespace td_ns;

    DispatchVectorBuilder<full_contig_fn_ptr_t, FullContigFactory, num_types>
        dvb;
    dvb.populate_dispatch_vector(full_contig_dispatch_vector);

    return;
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
