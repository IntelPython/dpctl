//=== POW.hpp -   Binary function POW                    ------  *-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of POW(x1, x2)
/// function.
//===---------------------------------------------------------------------===//

#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "kernels/dpctl_tensor_types.hpp"
#include "sycl_complex.hpp"
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
namespace pow
{

namespace td_ns = dpctl::tensor::type_dispatch;
namespace tu_ns = dpctl::tensor::type_utils;

template <typename argT1, typename argT2, typename resT> struct PowFunctor
{

    using supports_sg_loadstore = std::negation<
        std::disjunction<tu_ns::is_complex<argT1>, tu_ns::is_complex<argT2>>>;
    using supports_vec = std::negation<
        std::disjunction<tu_ns::is_complex<argT1>, tu_ns::is_complex<argT2>>>;

    resT operator()(const argT1 &in1, const argT2 &in2) const
    {
        if constexpr (std::is_integral_v<argT1> || std::is_integral_v<argT2>) {
            auto tmp1 = in1;
            auto tmp2 = in2;
            if constexpr (std::is_signed_v<argT2>) {
                if (tmp2 < 0) {
                    // invalid; return 0
                    return resT(0);
                }
            }
            resT res = 1;
            if (tmp1 == 1 || tmp2 == 0) {
                return res;
            }
            while (tmp2 > 0) {
                if (tmp2 & 1) {
                    res *= tmp1;
                }
                tmp2 >>= 1;
                tmp1 *= tmp1;
            }
            return res;
        }
        else if constexpr (tu_ns::is_complex<argT1>::value &&
                           tu_ns::is_complex<argT2>::value)
        {
#ifdef USE_SYCL_FOR_COMPLEX_TYPES
            using realT1 = typename argT1::value_type;
            using realT2 = typename argT2::value_type;

            return exprm_ns::pow(exprm_ns::complex<realT1>(in1),
                                 exprm_ns::complex<realT2>(in2));
#else
            return std::pow(in1, in2);
#endif
        }
        else {
            return std::pow(in1, in2);
        }
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz>
    operator()(const sycl::vec<argT1, vec_sz> &in1,
               const sycl::vec<argT2, vec_sz> &in2) const
    {
        if constexpr (std::is_integral_v<argT1> || std::is_integral_v<argT2>) {
            sycl::vec<resT, vec_sz> res;
#pragma unroll
            for (int i = 0; i < vec_sz; ++i) {
                auto tmp1 = in1[i];
                auto tmp2 = in2[i];
                if constexpr (std::is_signed_v<argT2>) {
                    if (tmp2 < 0) {
                        // invalid; yield 0
                        res[i] = 0;
                        continue;
                    }
                }
                resT res_tmp = 1;
                if (tmp1 == 1 || tmp2 == 0) {
                    res[i] = res_tmp;
                    continue;
                }
                while (tmp2 > 0) {
                    if (tmp2 & 1) {
                        res_tmp *= tmp1;
                    }
                    tmp2 >>= 1;
                    tmp1 *= tmp1;
                }
                res[i] = res_tmp;
            }
            return res;
        }
        else {
            auto res = sycl::pow(in1, in2);
            if constexpr (std::is_same_v<resT,
                                         typename decltype(res)::element_type>)
            {
                return res;
            }
            else {
                using dpctl::tensor::type_utils::vec_cast;

                return vec_cast<resT, typename decltype(res)::element_type,
                                vec_sz>(res);
            }
        }
    }
};

template <typename argT1,
          typename argT2,
          typename resT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using PowContigFunctor =
    elementwise_common::BinaryContigFunctor<argT1,
                                            argT2,
                                            resT,
                                            PowFunctor<argT1, argT2, resT>,
                                            vec_sz,
                                            n_vecs,
                                            enable_sg_loadstore>;

template <typename argT1, typename argT2, typename resT, typename IndexerT>
using PowStridedFunctor =
    elementwise_common::BinaryStridedFunctor<argT1,
                                             argT2,
                                             resT,
                                             IndexerT,
                                             PowFunctor<argT1, argT2, resT>>;

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
sycl::event pow_contig_impl(sycl::queue &exec_q,
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
class pow_strided_kernel;

template <typename argTy1, typename argTy2>
sycl::event pow_strided_impl(sycl::queue &exec_q,
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
        argTy1, argTy2, PowOutputType, PowStridedFunctor, pow_strided_kernel>(
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

template <typename argT, typename resT> struct PowInplaceFunctor
{

    using supports_sg_loadstore = std::negation<
        std::disjunction<tu_ns::is_complex<argT>, tu_ns::is_complex<resT>>>;
    using supports_vec = std::negation<
        std::disjunction<tu_ns::is_complex<argT>, tu_ns::is_complex<resT>>>;

    void operator()(resT &res, const argT &in)
    {
        if constexpr (std::is_integral_v<argT> || std::is_integral_v<resT>) {
            auto tmp1 = res;
            auto tmp2 = in;
            if constexpr (std::is_signed_v<argT>) {
                if (tmp2 < 0) {
                    // invalid; return 0
                    res = 0;
                    return;
                }
            }
            if (tmp1 == 1) {
                return;
            }
            if (tmp2 == 0) {
                res = 1;
                return;
            }
            resT res_tmp = 1;
            while (tmp2 > 0) {
                if (tmp2 & 1) {
                    res_tmp *= tmp1;
                }
                tmp2 >>= 1;
                tmp1 *= tmp1;
            }
            res = res_tmp;
        }
        else if constexpr (tu_ns::is_complex<argT>::value &&
                           tu_ns::is_complex<resT>::value)
        {
#ifdef USE_SYCL_FOR_COMPLEX_TYPES
            using r_resT = typename resT::value_type;
            using r_argT = typename argT::value_type;

            res = exprm_ns::pow(exprm_ns::complex<r_resT>(res),
                                exprm_ns::complex<r_argT>(in));
#else
            res = std::pow(res, in);
#endif
        }
        else {
            res = std::pow(res, in);
        }
        return;
    }

    template <int vec_sz>
    void operator()(sycl::vec<resT, vec_sz> &res,
                    const sycl::vec<argT, vec_sz> &in)
    {
        if constexpr (std::is_integral_v<argT> || std::is_integral_v<resT>) {
#pragma unroll
            for (int i = 0; i < vec_sz; ++i) {
                auto tmp1 = res[i];
                auto tmp2 = in[i];
                if constexpr (std::is_signed_v<argT>) {
                    if (tmp2 < 0) {
                        // invalid; return 0
                        res[i] = 0;
                        continue;
                    }
                }
                if (tmp1 == 1) {
                    continue;
                }
                if (tmp2 == 0) {
                    res[i] = 1;
                    continue;
                }
                resT res_tmp = 1;
                while (tmp2 > 0) {
                    if (tmp2 & 1) {
                        res_tmp *= tmp1;
                    }
                    tmp2 >>= 1;
                    tmp1 *= tmp1;
                }
                res[i] = res_tmp;
            }
        }
        else {
            res = sycl::pow(res, in);
        }
    }
};

template <typename argT,
          typename resT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using PowInplaceContigFunctor = elementwise_common::BinaryInplaceContigFunctor<
    argT,
    resT,
    PowInplaceFunctor<argT, resT>,
    vec_sz,
    n_vecs,
    enable_sg_loadstore>;

template <typename argT, typename resT, typename IndexerT>
using PowInplaceStridedFunctor =
    elementwise_common::BinaryInplaceStridedFunctor<
        argT,
        resT,
        IndexerT,
        PowInplaceFunctor<argT, resT>>;

template <typename argT,
          typename resT,
          unsigned int vec_sz,
          unsigned int n_vecs>
class pow_inplace_contig_kernel;

template <typename argTy, typename resTy>
sycl::event
pow_inplace_contig_impl(sycl::queue &exec_q,
                        size_t nelems,
                        const char *arg_p,
                        ssize_t arg_offset,
                        char *res_p,
                        ssize_t res_offset,
                        const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::binary_inplace_contig_impl<
        argTy, resTy, PowInplaceContigFunctor, pow_inplace_contig_kernel>(
        exec_q, nelems, arg_p, arg_offset, res_p, res_offset, depends);
}

template <typename fnT, typename T1, typename T2> struct PowInplaceContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename PowOutputType<T1, T2>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = pow_inplace_contig_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename resT, typename argT, typename IndexerT>
class pow_inplace_strided_kernel;

template <typename argTy, typename resTy>
sycl::event
pow_inplace_strided_impl(sycl::queue &exec_q,
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
        argTy, resTy, PowInplaceStridedFunctor, pow_inplace_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T1, typename T2>
struct PowInplaceStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename PowOutputType<T1, T2>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = pow_inplace_strided_impl<T1, T2>;
            return fn;
        }
    }
};

} // namespace pow
} // namespace kernels
} // namespace tensor
} // namespace dpctl
