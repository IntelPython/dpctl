//=== ATAN2.hpp -   Binary function ATAN2  ------               *-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of ATAN2(x1, x2)
/// function.
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
namespace atan2
{

namespace td_ns = dpctl::tensor::type_dispatch;
namespace tu_ns = dpctl::tensor::type_utils;

template <typename argT1, typename argT2, typename resT> struct Atan2Functor
{

    using supports_sg_loadstore = std::true_type;
    using supports_vec = std::false_type;

    resT operator()(const argT1 &in1, const argT2 &in2) const
    {
        if (std::isinf(in2) && !std::signbit(in2)) {
            if (std::isfinite(in1)) {
                return sycl::copysign(resT(0), in1);
            }
        }
        return std::atan2(in1, in2);
    }
};

template <typename argT1,
          typename argT2,
          typename resT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using Atan2ContigFunctor =
    elementwise_common::BinaryContigFunctor<argT1,
                                            argT2,
                                            resT,
                                            Atan2Functor<argT1, argT2, resT>,
                                            vec_sz,
                                            n_vecs,
                                            enable_sg_loadstore>;

template <typename argT1, typename argT2, typename resT, typename IndexerT>
using Atan2StridedFunctor =
    elementwise_common::BinaryStridedFunctor<argT1,
                                             argT2,
                                             resT,
                                             IndexerT,
                                             Atan2Functor<argT1, argT2, resT>>;

template <typename T1, typename T2> struct Atan2OutputType
{
    using value_type = typename std::disjunction< // disjunction is C++17
                                                  // feature, supported by DPC++
        td_ns::BinaryTypeMapResultEntry<T1,
                                        sycl::half,
                                        T2,
                                        sycl::half,
                                        sycl::half>,
        td_ns::BinaryTypeMapResultEntry<T1, float, T2, float, float>,
        td_ns::BinaryTypeMapResultEntry<T1, double, T2, double, double>,
        td_ns::DefaultResultEntry<void>>::result_type;
};

template <typename argT1,
          typename argT2,
          typename resT,
          unsigned int vec_sz,
          unsigned int n_vecs>
class atan2_contig_kernel;

template <typename argTy1, typename argTy2>
sycl::event atan2_contig_impl(sycl::queue &exec_q,
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
        argTy1, argTy2, Atan2OutputType, Atan2ContigFunctor,
        atan2_contig_kernel>(exec_q, nelems, arg1_p, arg1_offset, arg2_p,
                             arg2_offset, res_p, res_offset, depends);
}

template <typename fnT, typename T1, typename T2> struct Atan2ContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename Atan2OutputType<T1, T2>::value_type, void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = atan2_contig_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2> struct Atan2TypeMapFactory
{
    /*! @brief get typeid for output type of std::hypot(T1 x, T2 y) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename Atan2OutputType<T1, T2>::value_type;
        ;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename resT, typename IndexerT>
class atan2_strided_kernel;

template <typename argTy1, typename argTy2>
sycl::event
atan2_strided_impl(sycl::queue &exec_q,
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
        argTy1, argTy2, Atan2OutputType, Atan2StridedFunctor,
        atan2_strided_kernel>(exec_q, nelems, nd, shape_and_strides, arg1_p,
                              arg1_offset, arg2_p, arg2_offset, res_p,
                              res_offset, depends, additional_depends);
}

template <typename fnT, typename T1, typename T2> struct Atan2StridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename Atan2OutputType<T1, T2>::value_type, void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = atan2_strided_impl<T1, T2>;
            return fn;
        }
    }
};

} // namespace atan2
} // namespace kernels
} // namespace tensor
} // namespace dpctl
