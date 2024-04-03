//=== abs.hpp -   Unary function ABS                     ------  *-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of ABS(x) function.
//===---------------------------------------------------------------------===//

#pragma once
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "cabs_impl.hpp"
#include "kernels/elementwise_functions/common.hpp"

#include "kernels/dpctl_tensor_types.hpp"
#include "utils/offset_utils.hpp"
#include "utils/type_dispatch_building.hpp"
#include "utils/type_utils.hpp"

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace abs
{

namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;

template <typename argT, typename resT> struct AbsFunctor
{

    using is_constant = typename std::false_type;
    // constexpr resT constant_value = resT{};
    using supports_vec = typename std::false_type;
    using supports_sg_loadstore = typename std::negation<
        std::disjunction<is_complex<resT>, is_complex<argT>>>;

    resT operator()(const argT &x) const
    {

        if constexpr (std::is_same_v<argT, bool> ||
                      (std::is_integral<argT>::value &&
                       std::is_unsigned<argT>::value))
        {
            static_assert(std::is_same_v<resT, argT>);
            return x;
        }
        else {
            if constexpr (is_complex<argT>::value) {
                return detail::cabs(x);
            }
            else if constexpr (std::is_same_v<argT, sycl::half> ||
                               std::is_floating_point_v<argT>)
            {
                return (std::signbit(x) ? -x : x);
            }
            else {
                return sycl::abs(x);
            }
        }
    }
};

template <typename argT,
          typename resT = argT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using AbsContigFunctor =
    elementwise_common::UnaryContigFunctor<argT,
                                           resT,
                                           AbsFunctor<argT, resT>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename T> struct AbsOutputType
{
    using value_type = typename std::disjunction< // disjunction is C++17
                                                  // feature, supported by DPC++
        td_ns::TypeMapResultEntry<T, bool>,
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
        td_ns::TypeMapResultEntry<T, std::complex<float>, float>,
        td_ns::TypeMapResultEntry<T, std::complex<double>, double>,
        td_ns::DefaultResultEntry<void>>::result_type;
};

template <typename T1, typename T2, unsigned int vec_sz, unsigned int n_vecs>
class abs_contig_kernel;

template <typename argTy>
sycl::event abs_contig_impl(sycl::queue &exec_q,
                            size_t nelems,
                            const char *arg_p,
                            char *res_p,
                            const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::unary_contig_impl<
        argTy, AbsOutputType, AbsContigFunctor, abs_contig_kernel>(
        exec_q, nelems, arg_p, res_p, depends);
}

template <typename fnT, typename T> struct AbsContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename AbsOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = abs_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T> struct AbsTypeMapFactory
{
    /*! @brief get typeid for output type of abs(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename AbsOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename argTy, typename resTy, typename IndexerT>
using AbsStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, AbsFunctor<argTy, resTy>>;

template <typename T1, typename T2, typename T3> class abs_strided_kernel;

template <typename argTy>
sycl::event abs_strided_impl(sycl::queue &exec_q,
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
        argTy, AbsOutputType, AbsStridedFunctor, abs_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T> struct AbsStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename AbsOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = abs_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace abs
} // namespace kernels
} // namespace tensor
} // namespace dpctl
