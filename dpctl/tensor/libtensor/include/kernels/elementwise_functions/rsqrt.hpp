//=== rsqrt.hpp -   Unary function RSQRT                   ------
//*-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of RSQRT(x)
/// function that computes the reciprocal square root.
//===---------------------------------------------------------------------===//

#pragma once
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>
#include <type_traits>

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
namespace rsqrt
{

namespace td_ns = dpctl::tensor::type_dispatch;

template <typename argT, typename resT> struct RsqrtFunctor
{

    // is function constant for given argT
    using is_constant = typename std::false_type;
    // constant value, if constant
    // constexpr resT constant_value = resT{};
    // is function defined for sycl::vec
    using supports_vec = typename std::false_type;
    // do both argTy and resTy support sugroup store/load operation
    using supports_sg_loadstore = typename std::true_type;

    resT operator()(const argT &in) const
    {
        return sycl::rsqrt(in);
    }
};

template <typename argTy,
          typename resTy = argTy,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using RsqrtContigFunctor =
    elementwise_common::UnaryContigFunctor<argTy,
                                           resTy,
                                           RsqrtFunctor<argTy, resTy>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename argTy, typename resTy, typename IndexerT>
using RsqrtStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, RsqrtFunctor<argTy, resTy>>;

template <typename T> struct RsqrtOutputType
{
    using value_type = typename std::disjunction< // disjunction is C++17
                                                  // feature, supported by DPC++
        td_ns::TypeMapResultEntry<T, sycl::half, sycl::half>,
        td_ns::TypeMapResultEntry<T, float, float>,
        td_ns::TypeMapResultEntry<T, double, double>,
        td_ns::DefaultResultEntry<void>>::result_type;
};

template <typename T1, typename T2, unsigned int vec_sz, unsigned int n_vecs>
class rsqrt_contig_kernel;

template <typename argTy>
sycl::event rsqrt_contig_impl(sycl::queue &exec_q,
                              size_t nelems,
                              const char *arg_p,
                              char *res_p,
                              const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::unary_contig_impl<
        argTy, RsqrtOutputType, RsqrtContigFunctor, rsqrt_contig_kernel>(
        exec_q, nelems, arg_p, res_p, depends);
}

template <typename fnT, typename T> struct RsqrtContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename RsqrtOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = rsqrt_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T> struct RsqrtTypeMapFactory
{
    /*! @brief get typeid for output type of sycl::rsqrt(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename RsqrtOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3> class rsqrt_strided_kernel;

template <typename argTy>
sycl::event
rsqrt_strided_impl(sycl::queue &exec_q,
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
        argTy, RsqrtOutputType, RsqrtStridedFunctor, rsqrt_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T> struct RsqrtStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename RsqrtOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = rsqrt_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace rsqrt
} // namespace kernels
} // namespace tensor
} // namespace dpctl
