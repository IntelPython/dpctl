//=== bitwise_invert.hpp -   Unary function bitwise_invert      *-C++-*--/===//
//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2024 Intel Corporation
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
/// This file defines kernels for elementwise evaluation of bitwise_invert(x)
/// function that inverts bits of binary representation of the argument.
//===---------------------------------------------------------------------===//

#pragma once
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <type_traits>

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
namespace bitwise_invert
{

namespace td_ns = dpctl::tensor::type_dispatch;
namespace tu_ns = dpctl::tensor::type_utils;

using dpctl::tensor::type_utils::vec_cast;

template <typename argT, typename resT> struct BitwiseInvertFunctor
{
    static_assert(std::is_same_v<argT, resT>);
    static_assert(std::is_integral_v<argT> || std::is_same_v<argT, bool>);

    using is_constant = typename std::false_type;
    // constexpr resT constant_value = resT{};
    using supports_vec = typename std::negation<std::is_same<argT, bool>>;
    using supports_sg_loadstore = typename std::true_type;

    resT operator()(const argT &in) const
    {
        if constexpr (std::is_same_v<argT, bool>) {
            return !in;
        }
        else {
            return ~in;
        }
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz> operator()(const sycl::vec<argT, vec_sz> &in) const
    {
        return ~in;
    }
};

template <typename argT,
          typename resT = argT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using BitwiseInvertContigFunctor =
    elementwise_common::UnaryContigFunctor<argT,
                                           resT,
                                           BitwiseInvertFunctor<argT, resT>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename argTy, typename resTy, typename IndexerT>
using BitwiseInvertStridedFunctor =
    elementwise_common::UnaryStridedFunctor<argTy,
                                            resTy,
                                            IndexerT,
                                            BitwiseInvertFunctor<argTy, resTy>>;

template <typename argTy> struct BitwiseInvertOutputType
{
    using value_type = typename std::disjunction< // disjunction is C++17
                                                  // feature, supported by DPC++
        td_ns::TypeMapResultEntry<argTy, bool>,
        td_ns::TypeMapResultEntry<argTy, std::uint8_t>,
        td_ns::TypeMapResultEntry<argTy, std::uint16_t>,
        td_ns::TypeMapResultEntry<argTy, std::uint32_t>,
        td_ns::TypeMapResultEntry<argTy, std::uint64_t>,
        td_ns::TypeMapResultEntry<argTy, std::int8_t>,
        td_ns::TypeMapResultEntry<argTy, std::int16_t>,
        td_ns::TypeMapResultEntry<argTy, std::int32_t>,
        td_ns::TypeMapResultEntry<argTy, std::int64_t>,
        td_ns::DefaultResultEntry<void>>::result_type;
};

template <typename T1, typename T2, unsigned int vec_sz, unsigned int n_vecs>
class bitwise_invert_contig_kernel;

template <typename argTy>
sycl::event
bitwise_invert_contig_impl(sycl::queue &exec_q,
                           size_t nelems,
                           const char *arg_p,
                           char *res_p,
                           const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::unary_contig_impl<argTy, BitwiseInvertOutputType,
                                                 BitwiseInvertContigFunctor,
                                                 bitwise_invert_contig_kernel>(
        exec_q, nelems, arg_p, res_p, depends);
}

template <typename fnT, typename T> struct BitwiseInvertContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename BitwiseInvertOutputType<T>::value_type,
                          void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = bitwise_invert_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T> struct BitwiseInvertTypeMapFactory
{
    /*! @brief get typeid for output type of sycl::logical_not(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename BitwiseInvertOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3>
class bitwise_invert_strided_kernel;

template <typename argTy>
sycl::event
bitwise_invert_strided_impl(sycl::queue &exec_q,
                            size_t nelems,
                            int nd,
                            const ssize_t *shape_and_strides,
                            const char *arg_p,
                            ssize_t arg_offset,
                            char *res_p,
                            ssize_t res_offset,
                            const std::vector<sycl::event> &depends,
                            const std::vector<sycl::event> &additional_depends)
{
    return elementwise_common::unary_strided_impl<
        argTy, BitwiseInvertOutputType, BitwiseInvertStridedFunctor,
        bitwise_invert_strided_kernel>(exec_q, nelems, nd, shape_and_strides,
                                       arg_p, arg_offset, res_p, res_offset,
                                       depends, additional_depends);
}

template <typename fnT, typename T> struct BitwiseInvertStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename BitwiseInvertOutputType<T>::value_type,
                          void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = bitwise_invert_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace bitwise_invert
} // namespace kernels
} // namespace tensor
} // namespace dpctl
