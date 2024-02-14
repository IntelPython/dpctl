//=== true_divide.hpp -   Binary function DIVIDE         ------  *-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of DIVIDE(x1, x2)
/// function.
//===---------------------------------------------------------------------===//

#pragma once
#include <cstddef>
#include <cstdint>
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
namespace true_divide
{

namespace td_ns = dpctl::tensor::type_dispatch;
namespace tu_ns = dpctl::tensor::type_utils;

template <typename argT1, typename argT2, typename resT>
struct TrueDivideFunctor
{

    using supports_sg_loadstore = std::negation<
        std::disjunction<tu_ns::is_complex<argT1>, tu_ns::is_complex<argT2>>>;
    using supports_vec = std::negation<
        std::disjunction<tu_ns::is_complex<argT1>, tu_ns::is_complex<argT2>>>;

    resT operator()(const argT1 &in1, const argT2 &in2) const
    {
        if constexpr (tu_ns::is_complex<argT1>::value &&
                      tu_ns::is_complex<argT2>::value)
        {
#ifdef USE_SYCL_FOR_COMPLEX_TYPES
            using realT1 = typename argT1::value_type;
            using realT2 = typename argT2::value_type;

            return exprm_ns::complex<realT1>(in1) /
                   exprm_ns::complex<realT2>(in2);
#else
            return in1 / in2;
#endif
        }
        else if constexpr (tu_ns::is_complex<argT1>::value &&
                           !tu_ns::is_complex<argT2>::value)
        {
#ifdef USE_SYCL_FOR_COMPLEX_TYPES
            using realT1 = typename argT1::value_type;

            return exprm_ns::complex<realT1>(in1) / in2;
#else
            return in1 / in2;
#endif
        }
        else if constexpr (!tu_ns::is_complex<argT1>::value &&
                           tu_ns::is_complex<argT2>::value)
        {
#ifdef USE_SYCL_FOR_COMPLEX_TYPES
            using realT2 = typename argT2::value_type;

            return in1 / exprm_ns::complex<realT2>(in2);
#else
            return in1 / in2;
#endif
        }
        else {
            return in1 / in2;
        }
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz>
    operator()(const sycl::vec<argT1, vec_sz> &in1,
               const sycl::vec<argT2, vec_sz> &in2) const
    {
        auto tmp = in1 / in2;
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
using TrueDivideContigFunctor = elementwise_common::BinaryContigFunctor<
    argT1,
    argT2,
    resT,
    TrueDivideFunctor<argT1, argT2, resT>,
    vec_sz,
    n_vecs,
    enable_sg_loadstore>;

template <typename argT1, typename argT2, typename resT, typename IndexerT>
using TrueDivideStridedFunctor = elementwise_common::BinaryStridedFunctor<
    argT1,
    argT2,
    resT,
    IndexerT,
    TrueDivideFunctor<argT1, argT2, resT>>;

template <typename T1, typename T2> struct TrueDivideOutputType
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
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::complex<float>,
                                        T2,
                                        std::complex<float>,
                                        std::complex<float>>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::complex<float>,
                                        T2,
                                        float,
                                        std::complex<float>>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        float,
                                        T2,
                                        std::complex<float>,
                                        std::complex<float>>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::complex<double>,
                                        T2,
                                        std::complex<double>,
                                        std::complex<double>>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        double,
                                        T2,
                                        std::complex<double>,
                                        std::complex<double>>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::complex<double>,
                                        T2,
                                        double,
                                        std::complex<double>>,
        td_ns::DefaultResultEntry<void>>::result_type;
};

template <typename argT1,
          typename argT2,
          typename resT,
          unsigned int vec_sz,
          unsigned int n_vecs>
class true_divide_contig_kernel;

template <typename argTy1, typename argTy2>
sycl::event
true_divide_contig_impl(sycl::queue &exec_q,
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
        argTy1, argTy2, TrueDivideOutputType, TrueDivideContigFunctor,
        true_divide_contig_kernel>(exec_q, nelems, arg1_p, arg1_offset, arg2_p,
                                   arg2_offset, res_p, res_offset, depends);
}

template <typename fnT, typename T1, typename T2> struct TrueDivideContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename TrueDivideOutputType<T1, T2>::value_type,
                          void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = true_divide_contig_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2>
struct TrueDivideTypeMapFactory
{
    /*! @brief get typeid for output type of divide(T1 x, T2 y) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename TrueDivideOutputType<T1, T2>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename resT, typename IndexerT>
class true_divide_strided_kernel;

template <typename argTy1, typename argTy2>
sycl::event
true_divide_strided_impl(sycl::queue &exec_q,
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
        argTy1, argTy2, TrueDivideOutputType, TrueDivideStridedFunctor,
        true_divide_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg1_p, arg1_offset, arg2_p,
        arg2_offset, res_p, res_offset, depends, additional_depends);
}

template <typename fnT, typename T1, typename T2>
struct TrueDivideStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename TrueDivideOutputType<T1, T2>::value_type,
                          void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = true_divide_strided_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename argT1, typename argT2, typename resT>
using TrueDivideContigMatrixContigRowBroadcastingFunctor =
    elementwise_common::BinaryContigMatrixContigRowBroadcastingFunctor<
        argT1,
        argT2,
        resT,
        TrueDivideFunctor<argT1, argT2, resT>>;

template <typename argT1, typename argT2, typename resT>
using TrueDivideContigRowContigMatrixBroadcastingFunctor =
    elementwise_common::BinaryContigRowContigMatrixBroadcastingFunctor<
        argT1,
        argT2,
        resT,
        TrueDivideFunctor<argT1, argT2, resT>>;

template <typename argT1, typename argT2, typename resT>
class true_divide_matrix_row_broadcast_sg_krn;

template <typename argT1, typename argT2, typename resT>
class true_divide_row_matrix_broadcast_sg_krn;

template <typename argT1, typename argT2, typename resT>
sycl::event true_divide_contig_matrix_contig_row_broadcast_impl(
    sycl::queue &exec_q,
    std::vector<sycl::event> &host_tasks,
    size_t n0,
    size_t n1,
    const char *mat_p, // typeless pointer to (n0, n1) C-contiguous matrix
    ssize_t mat_offset,
    const char *vec_p, // typeless pointer to (n1,) contiguous row
    ssize_t vec_offset,
    char *res_p, // typeless pointer to (n0, n1) result C-contig. matrix,
                 //    res[i,j] = mat[i,j] / vec[j]
    ssize_t res_offset,
    const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::binary_contig_matrix_contig_row_broadcast_impl<
        argT1, argT2, resT, TrueDivideContigMatrixContigRowBroadcastingFunctor,
        true_divide_matrix_row_broadcast_sg_krn>(
        exec_q, host_tasks, n0, n1, mat_p, mat_offset, vec_p, vec_offset, res_p,
        res_offset, depends);
}

template <typename fnT, typename T1, typename T2>
struct TrueDivideContigMatrixContigRowBroadcastFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename TrueDivideOutputType<T1, T2>::value_type,
                          void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            using resT = typename TrueDivideOutputType<T1, T2>::value_type;
            if constexpr (dpctl::tensor::type_utils::is_complex<T1>::value ||
                          dpctl::tensor::type_utils::is_complex<T2>::value ||
                          dpctl::tensor::type_utils::is_complex<resT>::value)
            {
                fnT fn = nullptr;
                return fn;
            }
            else {
                fnT fn =
                    true_divide_contig_matrix_contig_row_broadcast_impl<T1, T2,
                                                                        resT>;
                return fn;
            }
        }
    }
};

template <typename argT1, typename argT2, typename resT>
sycl::event true_divide_contig_row_contig_matrix_broadcast_impl(
    sycl::queue &exec_q,
    std::vector<sycl::event> &host_tasks,
    size_t n0,
    size_t n1,
    const char *vec_p, // typeless pointer to (n1,) contiguous row
    ssize_t vec_offset,
    const char *mat_p, // typeless pointer to (n0, n1) C-contiguous matrix
    ssize_t mat_offset,
    char *res_p, // typeless pointer to (n0, n1) result C-contig. matrix,
                 //    res[i,j] = mat[i,j] + vec[j]
    ssize_t res_offset,
    const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::binary_contig_row_contig_matrix_broadcast_impl<
        argT1, argT2, resT, TrueDivideContigRowContigMatrixBroadcastingFunctor,
        true_divide_row_matrix_broadcast_sg_krn>(
        exec_q, host_tasks, n0, n1, vec_p, vec_offset, mat_p, mat_offset, res_p,
        res_offset, depends);
};

template <typename fnT, typename T1, typename T2>
struct TrueDivideContigRowContigMatrixBroadcastFactory
{
    fnT get()
    {
        using resT = typename TrueDivideOutputType<T1, T2>::value_type;
        if constexpr (std::is_same_v<resT, void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            if constexpr (dpctl::tensor::type_utils::is_complex<T1>::value ||
                          dpctl::tensor::type_utils::is_complex<T2>::value ||
                          dpctl::tensor::type_utils::is_complex<resT>::value)
            {
                fnT fn = nullptr;
                return fn;
            }
            else {
                fnT fn =
                    true_divide_contig_row_contig_matrix_broadcast_impl<T1, T2,
                                                                        resT>;
                return fn;
            }
        }
    }
};

template <typename argT, typename resT> struct TrueDivideInplaceFunctor
{

    using supports_sg_loadstore = std::negation<
        std::disjunction<tu_ns::is_complex<argT>, tu_ns::is_complex<resT>>>;
    using supports_vec = std::negation<
        std::disjunction<tu_ns::is_complex<argT>, tu_ns::is_complex<resT>>>;

    void operator()(resT &res, const argT &in)
    {
        if constexpr (tu_ns::is_complex<resT>::value) {
            if constexpr (tu_ns::is_complex<argT>::value) {
#ifdef USE_SYCL_FOR_COMPLEX_TYPES
                using res_rT = typename resT::value_type;
                using arg_rT = typename argT::value_type;

                auto res1 = exprm_ns::complex<res_rT>(res);
                res1 /= exprm_ns::complex<arg_rT>(in);
                res = res1;
#else
                res /= in;
#endif
            }
            else {
#ifdef USE_SYCL_FOR_COMPLEX_TYPES
                using res_rT = typename resT::value_type;

                auto res1 = exprm_ns::complex<res_rT>(res);
                res1 /= in;
                res = res1;
#else
                res /= in;
#endif
            }
        }
        else {
            res /= in;
        }
    }

    template <int vec_sz>
    void operator()(sycl::vec<resT, vec_sz> &res,
                    const sycl::vec<argT, vec_sz> &in)
    {
        res /= in;
    }
};

// cannot use the out of place table, as it permits real lhs and complex rhs
// T1 corresponds to the type of the rhs, while T2 corresponds to the lhs
// the type of the result must be the same as T2
template <typename T1, typename T2> struct TrueDivideInplaceOutputType
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
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::complex<float>,
                                        T2,
                                        std::complex<float>,
                                        std::complex<float>>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        float,
                                        T2,
                                        std::complex<float>,
                                        std::complex<float>>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::complex<double>,
                                        T2,
                                        std::complex<double>,
                                        std::complex<double>>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        double,
                                        T2,
                                        std::complex<double>,
                                        std::complex<double>>,
        td_ns::DefaultResultEntry<void>>::result_type;
};

template <typename fnT, typename T1, typename T2>
struct TrueDivideInplaceTypeMapFactory
{
    /*! @brief get typeid for output type of divide(T1 x, T2 y) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename TrueDivideInplaceOutputType<T1, T2>::value_type;
        static_assert(std::is_same_v<rT, T2> || std::is_same_v<rT, void>);
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename argT,
          typename resT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using TrueDivideInplaceContigFunctor =
    elementwise_common::BinaryInplaceContigFunctor<
        argT,
        resT,
        TrueDivideInplaceFunctor<argT, resT>,
        vec_sz,
        n_vecs,
        enable_sg_loadstore>;

template <typename argT, typename resT, typename IndexerT>
using TrueDivideInplaceStridedFunctor =
    elementwise_common::BinaryInplaceStridedFunctor<
        argT,
        resT,
        IndexerT,
        TrueDivideInplaceFunctor<argT, resT>>;

template <typename argT,
          typename resT,
          unsigned int vec_sz,
          unsigned int n_vecs>
class true_divide_inplace_contig_kernel;

template <typename argTy, typename resTy>
sycl::event
true_divide_inplace_contig_impl(sycl::queue &exec_q,
                                size_t nelems,
                                const char *arg_p,
                                ssize_t arg_offset,
                                char *res_p,
                                ssize_t res_offset,
                                const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::binary_inplace_contig_impl<
        argTy, resTy, TrueDivideInplaceContigFunctor,
        true_divide_inplace_contig_kernel>(exec_q, nelems, arg_p, arg_offset,
                                           res_p, res_offset, depends);
}

template <typename fnT, typename T1, typename T2>
struct TrueDivideInplaceContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename TrueDivideInplaceOutputType<
                                         T1, T2>::value_type,
                                     void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = true_divide_inplace_contig_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename resT, typename argT, typename IndexerT>
class true_divide_inplace_strided_kernel;

template <typename argTy, typename resTy>
sycl::event true_divide_inplace_strided_impl(
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
        argTy, resTy, TrueDivideInplaceStridedFunctor,
        true_divide_inplace_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T1, typename T2>
struct TrueDivideInplaceStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename TrueDivideInplaceOutputType<
                                         T1, T2>::value_type,
                                     void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = true_divide_inplace_strided_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename argT, typename resT>
class true_divide_inplace_row_matrix_broadcast_sg_krn;

template <typename argT, typename resT>
using TrueDivideInplaceRowMatrixBroadcastingFunctor =
    elementwise_common::BinaryInplaceRowMatrixBroadcastingFunctor<
        argT,
        resT,
        TrueDivideInplaceFunctor<argT, resT>>;

template <typename argT, typename resT>
sycl::event true_divide_inplace_row_matrix_broadcast_impl(
    sycl::queue &exec_q,
    std::vector<sycl::event> &host_tasks,
    size_t n0,
    size_t n1,
    const char *vec_p, // typeless pointer to (n1,) contiguous row
    ssize_t vec_offset,
    char *mat_p, // typeless pointer to (n0, n1) C-contiguous matrix
    ssize_t mat_offset,
    const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::binary_inplace_row_matrix_broadcast_impl<
        argT, resT, TrueDivideInplaceRowMatrixBroadcastingFunctor,
        true_divide_inplace_row_matrix_broadcast_sg_krn>(
        exec_q, host_tasks, n0, n1, vec_p, vec_offset, mat_p, mat_offset,
        depends);
}

template <typename fnT, typename T1, typename T2>
struct TrueDivideInplaceRowMatrixBroadcastFactory
{
    fnT get()
    {
        using resT = typename TrueDivideInplaceOutputType<T1, T2>::value_type;
        if constexpr (!std::is_same_v<resT, T2>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            if constexpr (dpctl::tensor::type_utils::is_complex<T1>::value ||
                          dpctl::tensor::type_utils::is_complex<T2>::value)
            {
                fnT fn = nullptr;
                return fn;
            }
            else {
                fnT fn = true_divide_inplace_row_matrix_broadcast_impl<T1, T2>;
                return fn;
            }
        }
    }
};

} // namespace true_divide
} // namespace kernels
} // namespace tensor
} // namespace dpctl
