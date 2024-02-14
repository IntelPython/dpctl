//=== bitwise_or.hpp -   Binary function BITWISE_OR    -------- *-C++-*--/===//
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
/// This file defines kernels for elementwise bitwise_or(ar1, ar2) operation.
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
#include "kernels/elementwise_functions/common_inplace.hpp"

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace bitwise_or
{

namespace td_ns = dpctl::tensor::type_dispatch;
namespace tu_ns = dpctl::tensor::type_utils;

template <typename argT1, typename argT2, typename resT> struct BitwiseOrFunctor
{
    static_assert(std::is_same_v<resT, argT1>);
    static_assert(std::is_same_v<resT, argT2>);

    using supports_sg_loadstore = typename std::true_type;
    using supports_vec = typename std::true_type;

    resT operator()(const argT1 &in1, const argT2 &in2) const
    {
        using tu_ns::convert_impl;

        if constexpr (std::is_same_v<resT, bool>) {
            return in1 || in2;
        }
        else {
            return (in1 | in2);
        }
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz>
    operator()(const sycl::vec<argT1, vec_sz> &in1,
               const sycl::vec<argT2, vec_sz> &in2) const
    {

        if constexpr (std::is_same_v<resT, bool>) {
            using dpctl::tensor::type_utils::vec_cast;

            auto tmp = (in1 || in2);
            return vec_cast<resT, typename decltype(tmp)::element_type, vec_sz>(
                tmp);
        }
        else {
            return (in1 | in2);
        }
    }
};

template <typename argT1,
          typename argT2,
          typename resT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using BitwiseOrContigFunctor = elementwise_common::BinaryContigFunctor<
    argT1,
    argT2,
    resT,
    BitwiseOrFunctor<argT1, argT2, resT>,
    vec_sz,
    n_vecs,
    enable_sg_loadstore>;

template <typename argT1, typename argT2, typename resT, typename IndexerT>
using BitwiseOrStridedFunctor = elementwise_common::BinaryStridedFunctor<
    argT1,
    argT2,
    resT,
    IndexerT,
    BitwiseOrFunctor<argT1, argT2, resT>>;

template <typename T1, typename T2> struct BitwiseOrOutputType
{
    using value_type = typename std::disjunction< // disjunction is C++17
                                                  // feature, supported by
                                                  // DPC++
        td_ns::BinaryTypeMapResultEntry<T1, bool, T2, bool, bool>,
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
        td_ns::DefaultResultEntry<void>>::result_type;
};

template <typename argT1,
          typename argT2,
          typename resT,
          unsigned int vec_sz,
          unsigned int n_vecs>
class bitwise_or_contig_kernel;

template <typename argTy1, typename argTy2>
sycl::event bitwise_or_contig_impl(sycl::queue &exec_q,
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
        argTy1, argTy2, BitwiseOrOutputType, BitwiseOrContigFunctor,
        bitwise_or_contig_kernel>(exec_q, nelems, arg1_p, arg1_offset, arg2_p,
                                  arg2_offset, res_p, res_offset, depends);
}

template <typename fnT, typename T1, typename T2> struct BitwiseOrContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename BitwiseOrOutputType<T1, T2>::value_type,
                          void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = bitwise_or_contig_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2> struct BitwiseOrTypeMapFactory
{
    /*! @brief get typeid for output type of operator()>(x, y), always bool
     */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename BitwiseOrOutputType<T1, T2>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename resT, typename IndexerT>
class bitwise_or_strided_kernel;

template <typename argTy1, typename argTy2>
sycl::event
bitwise_or_strided_impl(sycl::queue &exec_q,
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
        argTy1, argTy2, BitwiseOrOutputType, BitwiseOrStridedFunctor,
        bitwise_or_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg1_p, arg1_offset, arg2_p,
        arg2_offset, res_p, res_offset, depends, additional_depends);
}

template <typename fnT, typename T1, typename T2> struct BitwiseOrStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename BitwiseOrOutputType<T1, T2>::value_type,
                          void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = bitwise_or_strided_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename argT, typename resT> struct BitwiseOrInplaceFunctor
{
    using supports_sg_loadstore = typename std::true_type;
    using supports_vec = typename std::true_type;

    void operator()(resT &res, const argT &in) const
    {
        using tu_ns::convert_impl;

        if constexpr (std::is_same_v<resT, bool>) {
            res = res || in;
        }
        else {
            res |= in;
        }
    }

    template <int vec_sz>
    void operator()(sycl::vec<resT, vec_sz> &res,
                    const sycl::vec<argT, vec_sz> &in) const
    {

        if constexpr (std::is_same_v<resT, bool>) {
            using dpctl::tensor::type_utils::vec_cast;

            auto tmp = (res || in);
            res = vec_cast<resT, typename decltype(tmp)::element_type, vec_sz>(
                tmp);
        }
        else {
            res |= in;
        }
    }
};

template <typename argT,
          typename resT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using BitwiseOrInplaceContigFunctor =
    elementwise_common::BinaryInplaceContigFunctor<
        argT,
        resT,
        BitwiseOrInplaceFunctor<argT, resT>,
        vec_sz,
        n_vecs,
        enable_sg_loadstore>;

template <typename argT, typename resT, typename IndexerT>
using BitwiseOrInplaceStridedFunctor =
    elementwise_common::BinaryInplaceStridedFunctor<
        argT,
        resT,
        IndexerT,
        BitwiseOrInplaceFunctor<argT, resT>>;

template <typename argT,
          typename resT,
          unsigned int vec_sz,
          unsigned int n_vecs>
class bitwise_or_inplace_contig_kernel;

template <typename argTy, typename resTy>
sycl::event
bitwise_or_inplace_contig_impl(sycl::queue &exec_q,
                               size_t nelems,
                               const char *arg_p,
                               ssize_t arg_offset,
                               char *res_p,
                               ssize_t res_offset,
                               const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::binary_inplace_contig_impl<
        argTy, resTy, BitwiseOrInplaceContigFunctor,
        bitwise_or_inplace_contig_kernel>(exec_q, nelems, arg_p, arg_offset,
                                          res_p, res_offset, depends);
}

template <typename fnT, typename T1, typename T2>
struct BitwiseOrInplaceContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename BitwiseOrOutputType<T1, T2>::value_type,
                          void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = bitwise_or_inplace_contig_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename resT, typename argT, typename IndexerT>
class bitwise_or_inplace_strided_kernel;

template <typename argTy, typename resTy>
sycl::event bitwise_or_inplace_strided_impl(
    sycl::queue &exec_q,
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
    return elementwise_common::binary_inplace_strided_impl<
        argTy, resTy, BitwiseOrInplaceStridedFunctor,
        bitwise_or_inplace_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T1, typename T2>
struct BitwiseOrInplaceStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename BitwiseOrOutputType<T1, T2>::value_type,
                          void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = bitwise_or_inplace_strided_impl<T1, T2>;
            return fn;
        }
    }
};

} // namespace bitwise_or
} // namespace kernels
} // namespace tensor
} // namespace dpctl
