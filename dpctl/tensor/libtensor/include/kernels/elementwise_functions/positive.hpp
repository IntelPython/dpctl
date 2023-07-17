//=== positive.hpp -   Unary function POSITIVE           ------  *-C++-*--/===//
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
//===---------------------------------------------------------------------===//
///
/// \file
/// This file defines kernels for elementwise evaluation of POSITIVE(x)
/// function that returns x.
//===---------------------------------------------------------------------===//

#pragma once
#include <CL/sycl.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "kernels/elementwise_functions/common.hpp"

#include "utils/offset_utils.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"
#include <pybind11/pybind11.h>

#include <iostream>

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace positive
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;
using dpctl::tensor::type_utils::vec_cast;

template <typename argT, typename resT> struct PositiveFunctor
{

    using is_constant = typename std::false_type;
    // constexpr resT constant_value = resT{};
    using supports_vec = typename std::negation<
        std::disjunction<is_complex<resT>, is_complex<argT>>>;
    using supports_sg_loadstore = typename std::negation<
        std::disjunction<is_complex<resT>, is_complex<argT>>>;

    resT operator()(const argT &x)
    {
        return x;
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz> operator()(const sycl::vec<argT, vec_sz> &in)
    {
        auto const &res_vec = in;
        using deducedT = typename std::remove_cv_t<
            std::remove_reference_t<decltype(res_vec)>>::element_type;
        if constexpr (std::is_same_v<resT, deducedT>) {
            return res_vec;
        }
        else {
            return vec_cast<resT, deducedT, vec_sz>(res_vec);
        }
    }
};

template <typename argT,
          typename resT = argT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2>
using PositiveContigFunctor = elementwise_common::
    UnaryContigFunctor<argT, resT, PositiveFunctor<argT, resT>, vec_sz, n_vecs>;

template <typename T> struct PositiveOutputType
{
    using value_type = typename std::disjunction< // disjunction is C++17
                                                  // feature, supported by DPC++
        td_ns::TypeMapResultEntry<T, std::uint8_t>,
        td_ns::TypeMapResultEntry<T, std::uint16_t>,
        td_ns::TypeMapResultEntry<T, std::uint32_t>,
        td_ns::TypeMapResultEntry<T, std::uint64_t>,
        td_ns::TypeMapResultEntry<T, std::int8_t>,
        td_ns::TypeMapResultEntry<T, std::int16_t>,
        td_ns::TypeMapResultEntry<T, std::int32_t>,
        td_ns::TypeMapResultEntry<T, std::int64_t>,
        td_ns::TypeMapResultEntry<T, sycl::half>,
        td_ns::TypeMapResultEntry<T, float>,
        td_ns::TypeMapResultEntry<T, double>,
        td_ns::TypeMapResultEntry<T, std::complex<float>>,
        td_ns::TypeMapResultEntry<T, std::complex<double>>,
        td_ns::DefaultResultEntry<void>>::result_type;
};

template <typename T1, typename T2, unsigned int vec_sz, unsigned int n_vecs>
class positive_contig_kernel;

typedef sycl::event (*positive_contig_impl_fn_ptr_t)(
    sycl::queue,
    size_t,
    const char *,
    char *,
    const std::vector<sycl::event> &);

template <typename argTy>
sycl::event positive_contig_impl(sycl::queue exec_q,
                                 size_t nelems,
                                 const char *arg_p,
                                 char *res_p,
                                 const std::vector<sycl::event> &depends = {})
{
    sycl::event positive_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        size_t lws = 64;
        constexpr unsigned int vec_sz = 4;
        constexpr unsigned int n_vecs = 2;
        const size_t n_groups =
            ((nelems + lws * n_vecs * vec_sz - 1) / (lws * n_vecs * vec_sz));
        const auto gws_range = sycl::range<1>(n_groups * lws);
        const auto lws_range = sycl::range<1>(lws);

        using resTy = typename PositiveOutputType<argTy>::value_type;
        const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_p);
        resTy *res_tp = reinterpret_cast<resTy *>(res_p);

        cgh.parallel_for<positive_contig_kernel<argTy, resTy, vec_sz, n_vecs>>(
            sycl::nd_range<1>(gws_range, lws_range),
            PositiveContigFunctor<argTy, resTy, vec_sz, n_vecs>(arg_tp, res_tp,
                                                                nelems));
    });
    return positive_ev;
}

template <typename fnT, typename T> struct PositiveContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename PositiveOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = positive_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T> struct PositiveTypeMapFactory
{
    /*! @brief get typeid for output type of std::positive(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename PositiveOutputType<T>::value_type;
        ;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename argTy, typename resTy, typename IndexerT>
using PositiveStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, PositiveFunctor<argTy, resTy>>;

template <typename T1, typename T2, typename T3> class positive_strided_kernel;

typedef sycl::event (*positive_strided_impl_fn_ptr_t)(
    sycl::queue,
    size_t,
    int,
    const py::ssize_t *,
    const char *,
    py::ssize_t,
    char *,
    py::ssize_t,
    const std::vector<sycl::event> &,
    const std::vector<sycl::event> &);

template <typename argTy>
sycl::event
positive_strided_impl(sycl::queue exec_q,
                      size_t nelems,
                      int nd,
                      const py::ssize_t *shape_and_strides,
                      const char *arg_p,
                      py::ssize_t arg_offset,
                      char *res_p,
                      py::ssize_t res_offset,
                      const std::vector<sycl::event> &depends,
                      const std::vector<sycl::event> &additional_depends)
{
    sycl::event positive_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.depends_on(additional_depends);

        using resTy = typename PositiveOutputType<argTy>::value_type;
        using IndexerT =
            typename dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;

        IndexerT indexer{nd, arg_offset, res_offset, shape_and_strides};

        const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_p);
        resTy *res_tp = reinterpret_cast<resTy *>(res_p);

        cgh.parallel_for<positive_strided_kernel<argTy, resTy, IndexerT>>(
            {nelems}, PositiveStridedFunctor<argTy, resTy, IndexerT>(
                          arg_tp, res_tp, indexer));
    });
    return positive_ev;
}

template <typename fnT, typename T> struct PositiveStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename PositiveOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = positive_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace positive
} // namespace kernels
} // namespace tensor
} // namespace dpctl
