//=== remainder.hpp -   Binary function REMAINDER                ------
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
/// This file defines kernels for elementwise evaluation of the
/// modulo of tensor elements.
//===---------------------------------------------------------------------===//

#pragma once
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "kernels/dpctl_tensor_types.hpp"
#include "utils/offset_utils.hpp"
#include "utils/type_dispatch_building.hpp"
#include "utils/type_utils.hpp"

#include "kernels/elementwise_functions/common.hpp"
#include "kernels/elementwise_functions/common_inplace.hpp"

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace remainder
{

namespace td_ns = dpctl::tensor::type_dispatch;
namespace tu_ns = dpctl::tensor::type_utils;

template <typename argT1, typename argT2, typename resT> struct RemainderFunctor
{
    static_assert(std::is_same_v<argT1, argT2>);
    using supports_sg_loadstore = std::true_type;
    using supports_vec = std::true_type;

    resT operator()(const argT1 &in1, const argT2 &in2) const
    {
        if constexpr (std::is_integral_v<argT1> || std::is_integral_v<argT2>) {
            if (in2 == argT2(0)) {
                return resT(0);
            }
            if constexpr (std::is_signed_v<argT1> || std::is_signed_v<argT2>) {
                auto out = (in1 % in2);
                if (out != 0 && l_xor(in1 < 0, in2 < 0)) {
                    out += in2;
                }
                return out;
            }
            else {
                return (in1 % in2);
            }
        }
        else {
            auto rem = sycl::fmod(in1, in2);
            if (rem) {
                if (l_xor(in2 < 0, rem < 0)) {
                    rem += in2;
                }
            }
            else {
                rem = sycl::copysign(resT(0), in2);
            }
            return rem;
        }
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz>
    operator()(const sycl::vec<argT1, vec_sz> &in1,
               const sycl::vec<argT2, vec_sz> &in2) const
    {
        if constexpr (std::is_integral_v<argT1> || std::is_integral_v<argT2>) {
            sycl::vec<resT, vec_sz> rem;
#pragma unroll
            for (auto i = 0; i < vec_sz; ++i) {
                if (in2[i] == argT2(0)) {
                    rem[i] = resT(0);
                }
                else {
                    rem[i] = in1[i] % in2[i];
                    if constexpr (std::is_signed_v<argT1> ||
                                  std::is_signed_v<argT2>) {
                        if (rem[i] != 0 && l_xor(in1[i] < 0, in2[i] < 0)) {
                            rem[i] += in2[i];
                        }
                    }
                }
            }
            return rem;
        }
        else {
            auto rem = sycl::fmod(in1, in2);
            using remT = typename decltype(rem)::element_type;
#pragma unroll
            for (auto i = 0; i < vec_sz; ++i) {
                if (rem[i]) {
                    if (l_xor(in2[i] < 0, rem[i] < 0)) {
                        rem[i] += in2[i];
                    }
                }
                else {
                    rem[i] = sycl::copysign(remT(0), in2[i]);
                }
            }
            if constexpr (std::is_same_v<resT, remT>) {
                return rem;
            }
            else {
                using dpctl::tensor::type_utils::vec_cast;

                return vec_cast<resT, remT, vec_sz>(rem);
            }
        }
    }

private:
    bool l_xor(bool b1, bool b2) const
    {
        return (b1 != b2);
    }
};

template <typename argT1,
          typename argT2,
          typename resT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using RemainderContigFunctor = elementwise_common::BinaryContigFunctor<
    argT1,
    argT2,
    resT,
    RemainderFunctor<argT1, argT2, resT>,
    vec_sz,
    n_vecs,
    enable_sg_loadstore>;

template <typename argT1, typename argT2, typename resT, typename IndexerT>
using RemainderStridedFunctor = elementwise_common::BinaryStridedFunctor<
    argT1,
    argT2,
    resT,
    IndexerT,
    RemainderFunctor<argT1, argT2, resT>>;

template <typename T1, typename T2> struct RemainderOutputType
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
        td_ns::DefaultResultEntry<void>>::result_type;
};

template <typename argT1,
          typename argT2,
          typename resT,
          unsigned int vec_sz,
          unsigned int n_vecs>
class remainder_contig_kernel;

template <typename argTy1, typename argTy2>
sycl::event remainder_contig_impl(sycl::queue &exec_q,
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
        argTy1, argTy2, RemainderOutputType, RemainderContigFunctor,
        remainder_contig_kernel>(exec_q, nelems, arg1_p, arg1_offset, arg2_p,
                                 arg2_offset, res_p, res_offset, depends);
}

template <typename fnT, typename T1, typename T2> struct RemainderContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename RemainderOutputType<T1, T2>::value_type,
                          void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = remainder_contig_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2> struct RemainderTypeMapFactory
{
    /*! @brief get typeid for output type of remainder(T x, T y) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename RemainderOutputType<T1, T2>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename resT, typename IndexerT>
class remainder_strided_kernel;

template <typename argTy1, typename argTy2>
sycl::event
remainder_strided_impl(sycl::queue &exec_q,
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
        argTy1, argTy2, RemainderOutputType, RemainderStridedFunctor,
        remainder_strided_kernel>(exec_q, nelems, nd, shape_and_strides, arg1_p,
                                  arg1_offset, arg2_p, arg2_offset, res_p,
                                  res_offset, depends, additional_depends);
}

template <typename fnT, typename T1, typename T2> struct RemainderStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename RemainderOutputType<T1, T2>::value_type,
                          void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = remainder_strided_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename argT, typename resT> struct RemainderInplaceFunctor
{

    using supports_sg_loadstore = std::true_type;
    using supports_vec = std::true_type;

    // functor is only well-defined when argT and resT are the same
    static_assert(std::is_same_v<argT, resT>);

    void operator()(resT &res, const argT &in)
    {
        if constexpr (std::is_integral_v<argT> || std::is_integral_v<resT>) {
            if (in == argT(0)) {
                res = 0;
                return;
            }
            if constexpr (std::is_signed_v<argT> || std::is_signed_v<resT>) {
                auto tmp = res;
                res %= in;
                if (res != resT(0) && l_xor(tmp < 0, in < 0)) {
                    res += in;
                }
            }
            else {
                res %= in;
            }
        }
        else {
            res = sycl::fmod(res, in);
            if (res) {
                if (l_xor(in < 0, res < 0)) {
                    res += in;
                }
            }
            else {
                res = sycl::copysign(resT(0), in);
            }
        }
    }

    template <int vec_sz>
    void operator()(sycl::vec<resT, vec_sz> &res,
                    const sycl::vec<argT, vec_sz> &in)
    {
        if constexpr (std::is_integral_v<argT> || std::is_integral_v<resT>) {
#pragma unroll
            for (auto i = 0; i < vec_sz; ++i) {
                if (in[i] == argT(0)) {
                    res[i] = 0;
                }
                else {
                    auto rem = res[i] % in[i];
                    if constexpr (std::is_signed_v<argT> ||
                                  std::is_signed_v<resT>) {
                        if (rem != 0 && l_xor(res[i] < 0, in[i] < 0)) {
                            rem += in[i];
                        }
                    }
                    res[i] = rem;
                }
            }
        }
        else {
            res = sycl::fmod(res, in);
#pragma unroll
            for (auto i = 0; i < vec_sz; ++i) {
                if (res[i]) {
                    if (l_xor(in[i] < 0, res[i] < 0)) {
                        res[i] += in[i];
                    }
                }
                else {
                    res[i] = sycl::copysign(resT(0), in[i]);
                }
            }
        }
    }

private:
    bool l_xor(bool b1, bool b2) const
    {
        return (b1 != b2);
    }
};

template <typename argT,
          typename resT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using RemainderInplaceContigFunctor =
    elementwise_common::BinaryInplaceContigFunctor<
        argT,
        resT,
        RemainderInplaceFunctor<argT, resT>,
        vec_sz,
        n_vecs,
        enable_sg_loadstore>;

template <typename argT, typename resT, typename IndexerT>
using RemainderInplaceStridedFunctor =
    elementwise_common::BinaryInplaceStridedFunctor<
        argT,
        resT,
        IndexerT,
        RemainderInplaceFunctor<argT, resT>>;

template <typename argT,
          typename resT,
          unsigned int vec_sz,
          unsigned int n_vecs>
class remainder_inplace_contig_kernel;

template <typename argTy, typename resTy>
sycl::event
remainder_inplace_contig_impl(sycl::queue &exec_q,
                              size_t nelems,
                              const char *arg_p,
                              ssize_t arg_offset,
                              char *res_p,
                              ssize_t res_offset,
                              const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::binary_inplace_contig_impl<
        argTy, resTy, RemainderInplaceContigFunctor,
        remainder_inplace_contig_kernel>(exec_q, nelems, arg_p, arg_offset,
                                         res_p, res_offset, depends);
}

template <typename fnT, typename T1, typename T2>
struct RemainderInplaceContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename RemainderOutputType<T1, T2>::value_type,
                          void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = remainder_inplace_contig_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename resT, typename argT, typename IndexerT>
class remainder_inplace_strided_kernel;

template <typename argTy, typename resTy>
sycl::event remainder_inplace_strided_impl(
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
        argTy, resTy, RemainderInplaceStridedFunctor,
        remainder_inplace_strided_kernel>(exec_q, nelems, nd, shape_and_strides,
                                          arg_p, arg_offset, res_p, res_offset,
                                          depends, additional_depends);
}

template <typename fnT, typename T1, typename T2>
struct RemainderInplaceStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename RemainderOutputType<T1, T2>::value_type,
                          void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = remainder_inplace_strided_impl<T1, T2>;
            return fn;
        }
    }
};

} // namespace remainder
} // namespace kernels
} // namespace tensor
} // namespace dpctl
