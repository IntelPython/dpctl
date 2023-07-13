//=== POW.hpp -   Binary function POW                    ------  *-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of POW(x1, x2)
/// function.
//===---------------------------------------------------------------------===//

#pragma once
#include <CL/sycl.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
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
namespace pow
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;
namespace tu_ns = dpctl::tensor::type_utils;

template <typename argT1, typename argT2, typename resT> struct PowFunctor
{

    using supports_sg_loadstore = std::negation<
        std::disjunction<tu_ns::is_complex<argT1>, tu_ns::is_complex<argT2>>>;
    using supports_vec =
        std::negation<std::disjunction<tu_ns::is_complex<argT1>,
                                       tu_ns::is_complex<argT2>,
                                       std::is_integral<argT1>,
                                       std::is_integral<argT2>>>;

    resT operator()(argT1 in1, argT2 in2)
    {
        if constexpr (std::is_integral_v<argT1> || std::is_integral_v<argT2>) {
            if constexpr (std::is_signed_v<argT2>) {
                if (in2 < 0) {
                    // invalid; return 0
                    return resT(0);
                }
            }
            resT res = 1;
            if (in1 == 1 || in2 == 0) {
                return res;
            }
            while (in2 > 0) {
                if (in2 & 1) {
                    res *= in1;
                }
                in2 >>= 1;
                in1 *= in1;
            }
            return res;
        }
        else {
            return std::pow(in1, in2);
        }
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz> operator()(const sycl::vec<argT1, vec_sz> &in1,
                                       const sycl::vec<argT2, vec_sz> &in2)
    {
        auto res = sycl::pow(in1, in2);
        if constexpr (std::is_same_v<resT,
                                     typename decltype(res)::element_type>) {
            return res;
        }
        else {
            using dpctl::tensor::type_utils::vec_cast;

            return vec_cast<resT, typename decltype(res)::element_type, vec_sz>(
                res);
        }
    }
};

template <typename argT1,
          typename argT2,
          typename resT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2>
using PowContigFunctor =
    elementwise_common::BinaryContigFunctor<argT1,
                                            argT2,
                                            resT,
                                            PowFunctor<argT1, argT2, resT>,
                                            vec_sz,
                                            n_vecs>;

template <typename argT1, typename argT2, typename resT, typename IndexerT>
using PowStridedFunctor =
    elementwise_common::BinaryStridedFunctor<argT1,
                                             argT2,
                                             resT,
                                             IndexerT,
                                             PowFunctor<argT1, argT2, resT>>;

// TODO: when type promotion logic is better defined,
// consider implementing overloads of std::pow that take
// integers for the exponents. Seem to give better accuracy in
// some cases (complex data especially)
template <typename T1, typename T2> struct PowOutputType
{
    using value_type = typename std::disjunction< // disjunction is C++17
                                                  // feature, supported by DPC++
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint8_t,
                                        T2,
                                        std::uint8_t,
                                        std::uint8_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::int8_t,
                                        T2,
                                        std::int8_t,
                                        std::int8_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint16_t,
                                        T2,
                                        std::uint16_t,
                                        std::uint16_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::int16_t,
                                        T2,
                                        std::int16_t,
                                        std::int16_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint32_t,
                                        T2,
                                        std::uint32_t,
                                        std::uint32_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::int32_t,
                                        T2,
                                        std::int32_t,
                                        std::int32_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint64_t,
                                        T2,
                                        std::uint64_t,
                                        std::uint64_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::int64_t,
                                        T2,
                                        std::int64_t,
                                        std::int64_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        sycl::half,
                                        T2,
                                        sycl::half,
                                        sycl::half>,
        td_ns::BinaryTypeMapResultEntry<T1, float, T2, float, float>,
        td_ns::BinaryTypeMapResultEntry<T1, double, T2, double, double>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::complex<float>,
                                        T2,
                                        std::complex<float>,
                                        std::complex<float>>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::complex<double>,
                                        T2,
                                        std::complex<double>,
                                        std::complex<double>>,
        td_ns::DefaultResultEntry<void>>::result_type;
};

template <typename argT1,
          typename argT2,
          typename resT,
          unsigned int vec_sz,
          unsigned int n_vecs>
class pow_contig_kernel;

template <typename argTy1, typename argTy2>
sycl::event pow_contig_impl(sycl::queue exec_q,
                            size_t nelems,
                            const char *arg1_p,
                            py::ssize_t arg1_offset,
                            const char *arg2_p,
                            py::ssize_t arg2_offset,
                            char *res_p,
                            py::ssize_t res_offset,
                            const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::binary_contig_impl<
        argTy1, argTy2, PowOutputType, PowContigFunctor, pow_contig_kernel>(
        exec_q, nelems, arg1_p, arg1_offset, arg2_p, arg2_offset, res_p,
        res_offset, depends);
}

template <typename fnT, typename T1, typename T2> struct PowContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename PowOutputType<T1, T2>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = pow_contig_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2> struct PowTypeMapFactory
{
    /*! @brief get typeid for output type of std::pow(T1 x, T2 y) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename PowOutputType<T1, T2>::value_type;
        ;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename resT, typename IndexerT>
class pow_strided_strided_kernel;

template <typename argTy1, typename argTy2>
sycl::event pow_strided_impl(sycl::queue exec_q,
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
    return elementwise_common::binary_strided_impl<
        argTy1, argTy2, PowOutputType, PowStridedFunctor,
        pow_strided_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg1_p, arg1_offset, arg2_p,
        arg2_offset, res_p, res_offset, depends, additional_depends);
}

template <typename fnT, typename T1, typename T2> struct PowStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename PowOutputType<T1, T2>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = pow_strided_impl<T1, T2>;
            return fn;
        }
    }
};

} // namespace pow
} // namespace kernels
} // namespace tensor
} // namespace dpctl
