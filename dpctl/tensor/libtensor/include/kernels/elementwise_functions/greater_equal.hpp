//=== greater_equal.hpp -   Binary function GREATER_EQUAL        ------
//*-C++-*--/===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain in1 copy of the License at
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
/// This file defines kernels for elementwise evaluation of comparison of
/// tensor elements.
//===---------------------------------------------------------------------===//

#pragma once
#include <CL/sycl.hpp>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "utils/offset_utils.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

#include "kernels/elementwise_functions/common.hpp"
#include <pybind11/pybind11.h>

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace greater_equal
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;
namespace tu_ns = dpctl::tensor::type_utils;

template <typename argT1, typename argT2, typename resT>
struct GreaterEqualFunctor
{
    static_assert(std::is_same_v<resT, bool>);

    using supports_sg_loadstore = std::negation<
        std::disjunction<tu_ns::is_complex<argT1>, tu_ns::is_complex<argT2>>>;
    using supports_vec = std::conjunction<
        std::is_same<argT1, argT2>,
        std::negation<std::disjunction<tu_ns::is_complex<argT1>,
                                       tu_ns::is_complex<argT2>>>>;

    resT operator()(const argT1 &in1, const argT2 &in2)
    {
        if constexpr (std::is_same_v<argT1, std::complex<float>> &&
                      std::is_same_v<argT2, float>)
        {
            float real1 = std::real(in1);
            return (real1 == in2) ? (std::imag(in1) >= 0.0f) : real1 >= in2;
        }
        else if constexpr (std::is_same_v<argT1, float> &&
                           std::is_same_v<argT2, std::complex<float>>)
        {
            float real2 = std::real(in2);
            return (in1 == real2) ? (0.0f >= std::imag(in2)) : in1 >= real2;
        }
        else if constexpr (tu_ns::is_complex<argT1>::value ||
                           tu_ns::is_complex<argT2>::value)
        {
            static_assert(std::is_same_v<argT1, argT2>);
            using realT = typename argT1::value_type;
            realT real1 = std::real(in1);
            realT real2 = std::real(in2);

            return (real1 == real2) ? (std::imag(in1) >= std::imag(in2))
                                    : real1 >= real2;
        }
        else {
            return (in1 >= in2);
        }
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz> operator()(const sycl::vec<argT1, vec_sz> &in1,
                                       const sycl::vec<argT2, vec_sz> &in2)
    {

        auto tmp = (in1 >= in2);

        if constexpr (std::is_same_v<resT,
                                     typename decltype(tmp)::element_type>) {
            return tmp;
        }
        else {
            using dpctl::tensor::type_utils::vec_cast;

            return vec_cast<resT, typename decltype(tmp)::element_type, vec_sz>(
                tmp);
        }
    }
};

template <typename argT1,
          typename argT2,
          typename resT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2>
using GreaterEqualContigFunctor = elementwise_common::BinaryContigFunctor<
    argT1,
    argT2,
    resT,
    GreaterEqualFunctor<argT1, argT2, resT>,
    vec_sz,
    n_vecs>;

template <typename argT1, typename argT2, typename resT, typename IndexerT>
using GreaterEqualStridedFunctor = elementwise_common::BinaryStridedFunctor<
    argT1,
    argT2,
    resT,
    IndexerT,
    GreaterEqualFunctor<argT1, argT2, resT>>;

template <typename T1, typename T2> struct GreaterEqualOutputType
{
    using value_type = typename std::disjunction< // disjunction is C++17
                                                  // feature, supported by DPC++
        td_ns::BinaryTypeMapResultEntry<T1, bool, T2, bool, bool>,
        td_ns::
            BinaryTypeMapResultEntry<T1, std::uint8_t, T2, std::uint8_t, bool>,
        td_ns::BinaryTypeMapResultEntry<T1, std::int8_t, T2, std::int8_t, bool>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint16_t,
                                        T2,
                                        std::uint16_t,
                                        bool>,
        td_ns::
            BinaryTypeMapResultEntry<T1, std::int16_t, T2, std::int16_t, bool>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint32_t,
                                        T2,
                                        std::uint32_t,
                                        bool>,
        td_ns::
            BinaryTypeMapResultEntry<T1, std::int32_t, T2, std::int32_t, bool>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint64_t,
                                        T2,
                                        std::uint64_t,
                                        bool>,
        td_ns::
            BinaryTypeMapResultEntry<T1, std::int64_t, T2, std::int64_t, bool>,
        td_ns::BinaryTypeMapResultEntry<T1, sycl::half, T2, sycl::half, bool>,
        td_ns::BinaryTypeMapResultEntry<T1, float, T2, float, bool>,
        td_ns::BinaryTypeMapResultEntry<T1, double, T2, double, bool>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::complex<float>,
                                        T2,
                                        std::complex<float>,
                                        bool>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::complex<double>,
                                        T2,
                                        std::complex<double>,
                                        bool>,
        td_ns::
            BinaryTypeMapResultEntry<T1, float, T2, std::complex<float>, bool>,
        td_ns::
            BinaryTypeMapResultEntry<T1, std::complex<float>, T2, float, bool>,
        td_ns::DefaultResultEntry<void>>::result_type;
};

template <typename argT1,
          typename argT2,
          typename resT,
          unsigned int vec_sz,
          unsigned int n_vecs>
class greater_equal_contig_kernel;

template <typename argTy1, typename argTy2>
sycl::event
greater_equal_contig_impl(sycl::queue exec_q,
                          size_t nelems,
                          const char *arg1_p,
                          py::ssize_t arg1_offset,
                          const char *arg2_p,
                          py::ssize_t arg2_offset,
                          char *res_p,
                          py::ssize_t res_offset,
                          const std::vector<sycl::event> &depends = {})
{
    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        size_t lws = 64;
        constexpr unsigned int vec_sz = 4;
        constexpr unsigned int n_vecs = 2;
        const size_t n_groups =
            ((nelems + lws * n_vecs * vec_sz - 1) / (lws * n_vecs * vec_sz));
        const auto gws_range = sycl::range<1>(n_groups * lws);
        const auto lws_range = sycl::range<1>(lws);

        using resTy =
            typename GreaterEqualOutputType<argTy1, argTy2>::value_type;

        const argTy1 *arg1_tp =
            reinterpret_cast<const argTy1 *>(arg1_p) + arg1_offset;
        const argTy2 *arg2_tp =
            reinterpret_cast<const argTy2 *>(arg2_p) + arg2_offset;
        resTy *res_tp = reinterpret_cast<resTy *>(res_p) + res_offset;

        cgh.parallel_for<
            greater_equal_contig_kernel<argTy1, argTy2, resTy, vec_sz, n_vecs>>(
            sycl::nd_range<1>(gws_range, lws_range),
            GreaterEqualContigFunctor<argTy1, argTy2, resTy, vec_sz, n_vecs>(
                arg1_tp, arg2_tp, res_tp, nelems));
    });
    return comp_ev;
}

template <typename fnT, typename T1, typename T2>
struct GreaterEqualContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename GreaterEqualOutputType<T1, T2>::value_type,
                          void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = greater_equal_contig_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2>
struct GreaterEqualTypeMapFactory
{
    /*! @brief get typeid for output type of operator()>(x, y), always bool */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename GreaterEqualOutputType<T1, T2>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename resT, typename IndexerT>
class greater_equal_strided_kernel;

template <typename argTy1, typename argTy2>
sycl::event
greater_equal_strided_impl(sycl::queue exec_q,
                           size_t nelems,
                           int nd,
                           const py::ssize_t *shape_and_strides,
                           const char *arg1_p,
                           py::ssize_t arg1_offset,
                           const char *arg2_p,
                           py::ssize_t arg2_offset,
                           char *res_p,
                           py::ssize_t res_offset,
                           const std::vector<sycl::event> &depends,
                           const std::vector<sycl::event> &additional_depends)
{
    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.depends_on(additional_depends);

        using resTy =
            typename GreaterEqualOutputType<argTy1, argTy2>::value_type;

        using IndexerT =
            typename dpctl::tensor::offset_utils::ThreeOffsets_StridedIndexer;

        IndexerT indexer{nd, arg1_offset, arg2_offset, res_offset,
                         shape_and_strides};

        const argTy1 *arg1_tp = reinterpret_cast<const argTy1 *>(arg1_p);
        const argTy2 *arg2_tp = reinterpret_cast<const argTy2 *>(arg2_p);
        resTy *res_tp = reinterpret_cast<resTy *>(res_p);

        cgh.parallel_for<
            greater_equal_strided_kernel<argTy1, argTy2, resTy, IndexerT>>(
            {nelems},
            GreaterEqualStridedFunctor<argTy1, argTy2, resTy, IndexerT>(
                arg1_tp, arg2_tp, res_tp, indexer));
    });
    return comp_ev;
}

template <typename fnT, typename T1, typename T2>
struct GreaterEqualStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename GreaterEqualOutputType<T1, T2>::value_type,
                          void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = greater_equal_strided_impl<T1, T2>;
            return fn;
        }
    }
};

} // namespace greater_equal
} // namespace kernels
} // namespace tensor
} // namespace dpctl
