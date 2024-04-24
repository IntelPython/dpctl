//=== greater.hpp -   Binary function GREATER              ------
//*-C++-*--/===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2024 Intel Corporation
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
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "utils/math_utils.hpp"
#include "utils/offset_utils.hpp"
#include "utils/type_dispatch_building.hpp"
#include "utils/type_utils.hpp"

#include "kernels/dpctl_tensor_types.hpp"
#include "kernels/elementwise_functions/common.hpp"

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace greater
{

namespace td_ns = dpctl::tensor::type_dispatch;
namespace tu_ns = dpctl::tensor::type_utils;

template <typename argT1, typename argT2, typename resT> struct GreaterFunctor
{
    static_assert(std::is_same_v<resT, bool>);

    using supports_sg_loadstore = std::negation<
        std::disjunction<tu_ns::is_complex<argT1>, tu_ns::is_complex<argT2>>>;
    using supports_vec = std::conjunction<
        std::is_same<argT1, argT2>,
        std::negation<std::disjunction<tu_ns::is_complex<argT1>,
                                       tu_ns::is_complex<argT2>>>>;

    resT operator()(const argT1 &in1, const argT2 &in2) const
    {
        if constexpr (tu_ns::is_complex<argT1>::value ||
                      tu_ns::is_complex<argT2>::value)
        {
            static_assert(std::is_same_v<argT1, argT2>);
            using dpctl::tensor::math_utils::greater_complex;
            return greater_complex<argT1>(in1, in2);
        }
        else {
            if constexpr (std::is_integral_v<argT1> &&
                          std::is_integral_v<argT2> &&
                          std::is_signed_v<argT1> != std::is_signed_v<argT2>)
            {
                if constexpr (std::is_signed_v<argT1> &&
                              !std::is_signed_v<argT2>) {
                    return (in1 < 0) ? false : (static_cast<argT2>(in1) > in2);
                }
                else {
                    if constexpr (!std::is_signed_v<argT1> &&
                                  std::is_signed_v<argT2>) {
                        return (in2 < 0) ? true
                                         : (in1 > static_cast<argT1>(in2));
                    }
                }
            }
            else {
                return (in1 > in2);
            }
        }
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz>
    operator()(const sycl::vec<argT1, vec_sz> &in1,
               const sycl::vec<argT2, vec_sz> &in2) const
    {

        auto tmp = (in1 > in2);

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
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using GreaterContigFunctor =
    elementwise_common::BinaryContigFunctor<argT1,
                                            argT2,
                                            resT,
                                            GreaterFunctor<argT1, argT2, resT>,
                                            vec_sz,
                                            n_vecs,
                                            enable_sg_loadstore>;

template <typename argT1, typename argT2, typename resT, typename IndexerT>
using GreaterStridedFunctor = elementwise_common::BinaryStridedFunctor<
    argT1,
    argT2,
    resT,
    IndexerT,
    GreaterFunctor<argT1, argT2, resT>>;

template <typename T1, typename T2> struct GreaterOutputType
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
        td_ns::
            BinaryTypeMapResultEntry<T1, std::uint64_t, T2, std::int64_t, bool>,
        td_ns::
            BinaryTypeMapResultEntry<T1, std::int64_t, T2, std::uint64_t, bool>,
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
        td_ns::DefaultResultEntry<void>>::result_type;
};

template <typename argT1,
          typename argT2,
          typename resT,
          unsigned int vec_sz,
          unsigned int n_vecs>
class greater_contig_kernel;

template <typename argTy1, typename argTy2>
sycl::event greater_contig_impl(sycl::queue &exec_q,
                                size_t nelems,
                                const char *arg1_p,
                                ssize_t arg1_offset,
                                const char *arg2_p,
                                ssize_t arg2_offset,
                                char *res_p,
                                ssize_t res_offset,
                                const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::binary_contig_impl<
        argTy1, argTy2, GreaterOutputType, GreaterContigFunctor,
        greater_contig_kernel>(exec_q, nelems, arg1_p, arg1_offset, arg2_p,
                               arg2_offset, res_p, res_offset, depends);
}

template <typename fnT, typename T1, typename T2> struct GreaterContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename GreaterOutputType<T1, T2>::value_type, void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = greater_contig_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2> struct GreaterTypeMapFactory
{
    /*! @brief get typeid for output type of operator()>(x, y), always bool */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename GreaterOutputType<T1, T2>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename resT, typename IndexerT>
class greater_strided_kernel;

template <typename argTy1, typename argTy2>
sycl::event
greater_strided_impl(sycl::queue &exec_q,
                     size_t nelems,
                     int nd,
                     const ssize_t *shape_and_strides,
                     const char *arg1_p,
                     ssize_t arg1_offset,
                     const char *arg2_p,
                     ssize_t arg2_offset,
                     char *res_p,
                     ssize_t res_offset,
                     const std::vector<sycl::event> &depends,
                     const std::vector<sycl::event> &additional_depends)
{
    return elementwise_common::binary_strided_impl<
        argTy1, argTy2, GreaterOutputType, GreaterStridedFunctor,
        greater_strided_kernel>(exec_q, nelems, nd, shape_and_strides, arg1_p,
                                arg1_offset, arg2_p, arg2_offset, res_p,
                                res_offset, depends, additional_depends);
}

template <typename fnT, typename T1, typename T2> struct GreaterStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename GreaterOutputType<T1, T2>::value_type, void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = greater_strided_impl<T1, T2>;
            return fn;
        }
    }
};

} // namespace greater
} // namespace kernels
} // namespace tensor
} // namespace dpctl
