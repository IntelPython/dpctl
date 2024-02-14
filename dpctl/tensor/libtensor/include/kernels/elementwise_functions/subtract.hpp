//=== subtract.hpp -   Binary function SUBTRACT         ------  *-C++-*--/===//
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
namespace subtract
{

namespace td_ns = dpctl::tensor::type_dispatch;
namespace tu_ns = dpctl::tensor::type_utils;

template <typename argT1, typename argT2, typename resT> struct SubtractFunctor
{

    using supports_sg_loadstore = std::negation<
        std::disjunction<tu_ns::is_complex<argT1>, tu_ns::is_complex<argT2>>>;
    using supports_vec = std::negation<
        std::disjunction<tu_ns::is_complex<argT1>, tu_ns::is_complex<argT2>>>;

    resT operator()(const argT1 &in1, const argT2 &in2) const
    {
        return in1 - in2;
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz>
    operator()(const sycl::vec<argT1, vec_sz> &in1,
               const sycl::vec<argT2, vec_sz> &in2) const
    {
        auto tmp = in1 - in2;
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
using SubtractContigFunctor =
    elementwise_common::BinaryContigFunctor<argT1,
                                            argT2,
                                            resT,
                                            SubtractFunctor<argT1, argT2, resT>,
                                            vec_sz,
                                            n_vecs,
                                            enable_sg_loadstore>;

template <typename argT1, typename argT2, typename resT, typename IndexerT>
using SubtractStridedFunctor = elementwise_common::BinaryStridedFunctor<
    argT1,
    argT2,
    resT,
    IndexerT,
    SubtractFunctor<argT1, argT2, resT>>;

template <typename T1, typename T2> struct SubtractOutputType
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
class subtract_contig_kernel;

template <typename argTy1, typename argTy2>
sycl::event subtract_contig_impl(sycl::queue &exec_q,
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
        argTy1, argTy2, SubtractOutputType, SubtractContigFunctor,
        subtract_contig_kernel>(exec_q, nelems, arg1_p, arg1_offset, arg2_p,
                                arg2_offset, res_p, res_offset, depends);
}

template <typename fnT, typename T1, typename T2> struct SubtractContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename SubtractOutputType<T1, T2>::value_type,
                          void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = subtract_contig_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2> struct SubtractTypeMapFactory
{
    /*! @brief get typeid for output type of divide(T1 x, T2 y) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename SubtractOutputType<T1, T2>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename resT, typename IndexerT>
class subtract_strided_kernel;

template <typename argTy1, typename argTy2>
sycl::event
subtract_strided_impl(sycl::queue &exec_q,
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
        argTy1, argTy2, SubtractOutputType, SubtractStridedFunctor,
        subtract_strided_kernel>(exec_q, nelems, nd, shape_and_strides, arg1_p,
                                 arg1_offset, arg2_p, arg2_offset, res_p,
                                 res_offset, depends, additional_depends);
}

template <typename fnT, typename T1, typename T2> struct SubtractStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename SubtractOutputType<T1, T2>::value_type,
                          void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = subtract_strided_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename argT1, typename argT2, typename resT>
using SubtractContigMatrixContigRowBroadcastingFunctor =
    elementwise_common::BinaryContigMatrixContigRowBroadcastingFunctor<
        argT1,
        argT2,
        resT,
        SubtractFunctor<argT1, argT2, resT>>;

template <typename argT1, typename argT2, typename resT>
using SubtractContigRowContigMatrixBroadcastingFunctor =
    elementwise_common::BinaryContigRowContigMatrixBroadcastingFunctor<
        argT1,
        argT2,
        resT,
        SubtractFunctor<argT1, argT2, resT>>;

template <typename argT1, typename argT2, typename resT>
class subtract_matrix_row_broadcast_sg_krn;

template <typename argT1, typename argT2, typename resT>
class subtract_row_matrix_broadcast_sg_krn;

template <typename argT1, typename argT2, typename resT>
sycl::event subtract_contig_matrix_contig_row_broadcast_impl(
    sycl::queue &exec_q,
    std::vector<sycl::event> &host_tasks,
    size_t n0,
    size_t n1,
    const char *mat_p, // typeless pointer to (n0, n1) C-contiguous matrix
    ssize_t mat_offset,
    const char *vec_p, // typeless pointer to (n1,) contiguous row
    ssize_t vec_offset,
    char *res_p, // typeless pointer to (n0, n1) result C-contig. matrix,
                 //    res[i,j] = mat[i,j] - vec[j]
    ssize_t res_offset,
    const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::binary_contig_matrix_contig_row_broadcast_impl<
        argT1, argT2, resT, SubtractContigMatrixContigRowBroadcastingFunctor,
        subtract_matrix_row_broadcast_sg_krn>(exec_q, host_tasks, n0, n1, mat_p,
                                              mat_offset, vec_p, vec_offset,
                                              res_p, res_offset, depends);
}

template <typename fnT, typename T1, typename T2>
struct SubtractContigMatrixContigRowBroadcastFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename SubtractOutputType<T1, T2>::value_type,
                          void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            using resT = typename SubtractOutputType<T1, T2>::value_type;
            if constexpr (dpctl::tensor::type_utils::is_complex<T1>::value ||
                          dpctl::tensor::type_utils::is_complex<T2>::value ||
                          dpctl::tensor::type_utils::is_complex<resT>::value)
            {
                fnT fn = nullptr;
                return fn;
            }
            else {
                fnT fn =
                    subtract_contig_matrix_contig_row_broadcast_impl<T1, T2,
                                                                     resT>;
                return fn;
            }
        }
    }
};

template <typename argT1, typename argT2, typename resT>
sycl::event subtract_contig_row_contig_matrix_broadcast_impl(
    sycl::queue &exec_q,
    std::vector<sycl::event> &host_tasks,
    size_t n0,
    size_t n1,
    const char *vec_p, // typeless pointer to (n1,) contiguous row
    ssize_t vec_offset,
    const char *mat_p, // typeless pointer to (n0, n1) C-contiguous matrix
    ssize_t mat_offset,
    char *res_p, // typeless pointer to (n0, n1) result C-contig. matrix,
                 //    res[i,j] = op(vec[j], mat[i,j])
    ssize_t res_offset,
    const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::binary_contig_row_contig_matrix_broadcast_impl<
        argT1, argT2, resT, SubtractContigRowContigMatrixBroadcastingFunctor,
        subtract_row_matrix_broadcast_sg_krn>(exec_q, host_tasks, n0, n1, vec_p,
                                              vec_offset, mat_p, mat_offset,
                                              res_p, res_offset, depends);
}

template <typename fnT, typename T1, typename T2>
struct SubtractContigRowContigMatrixBroadcastFactory
{
    fnT get()
    {
        using resT = typename SubtractOutputType<T1, T2>::value_type;
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
                    subtract_contig_row_contig_matrix_broadcast_impl<T1, T2,
                                                                     resT>;
                return fn;
            }
        }
    }
};

template <typename argT, typename resT> struct SubtractInplaceFunctor
{

    using supports_sg_loadstore = std::negation<
        std::disjunction<tu_ns::is_complex<argT>, tu_ns::is_complex<resT>>>;
    using supports_vec = std::negation<
        std::disjunction<tu_ns::is_complex<argT>, tu_ns::is_complex<resT>>>;

    void operator()(resT &res, const argT &in)
    {
        res -= in;
    }

    template <int vec_sz>
    void operator()(sycl::vec<resT, vec_sz> &res,
                    const sycl::vec<argT, vec_sz> &in)
    {
        res -= in;
    }
};

template <typename argT,
          typename resT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using SubtractInplaceContigFunctor =
    elementwise_common::BinaryInplaceContigFunctor<
        argT,
        resT,
        SubtractInplaceFunctor<argT, resT>,
        vec_sz,
        n_vecs,
        enable_sg_loadstore>;

template <typename argT, typename resT, typename IndexerT>
using SubtractInplaceStridedFunctor =
    elementwise_common::BinaryInplaceStridedFunctor<
        argT,
        resT,
        IndexerT,
        SubtractInplaceFunctor<argT, resT>>;

template <typename argT,
          typename resT,
          unsigned int vec_sz,
          unsigned int n_vecs>
class subtract_inplace_contig_kernel;

template <typename argTy, typename resTy>
sycl::event
subtract_inplace_contig_impl(sycl::queue &exec_q,
                             size_t nelems,
                             const char *arg_p,
                             ssize_t arg_offset,
                             char *res_p,
                             ssize_t res_offset,
                             const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::binary_inplace_contig_impl<
        argTy, resTy, SubtractInplaceContigFunctor,
        subtract_inplace_contig_kernel>(exec_q, nelems, arg_p, arg_offset,
                                        res_p, res_offset, depends);
}

template <typename fnT, typename T1, typename T2>
struct SubtractInplaceContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename SubtractOutputType<T1, T2>::value_type,
                          void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = subtract_inplace_contig_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename resT, typename argT, typename IndexerT>
class subtract_inplace_strided_kernel;

template <typename argTy, typename resTy>
sycl::event subtract_inplace_strided_impl(
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
        argTy, resTy, SubtractInplaceStridedFunctor,
        subtract_inplace_strided_kernel>(exec_q, nelems, nd, shape_and_strides,
                                         arg_p, arg_offset, res_p, res_offset,
                                         depends, additional_depends);
}

template <typename fnT, typename T1, typename T2>
struct SubtractInplaceStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename SubtractOutputType<T1, T2>::value_type,
                          void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = subtract_inplace_strided_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename argT, typename resT>
class subtract_inplace_row_matrix_broadcast_sg_krn;

template <typename argT, typename resT>
using SubtractInplaceRowMatrixBroadcastingFunctor =
    elementwise_common::BinaryInplaceRowMatrixBroadcastingFunctor<
        argT,
        resT,
        SubtractInplaceFunctor<argT, resT>>;

template <typename argT, typename resT>
sycl::event subtract_inplace_row_matrix_broadcast_impl(
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
        argT, resT, SubtractInplaceRowMatrixBroadcastingFunctor,
        subtract_inplace_row_matrix_broadcast_sg_krn>(
        exec_q, host_tasks, n0, n1, vec_p, vec_offset, mat_p, mat_offset,
        depends);
}

template <typename fnT, typename T1, typename T2>
struct SubtractInplaceRowMatrixBroadcastFactory
{
    fnT get()
    {
        using resT = typename SubtractOutputType<T1, T2>::value_type;
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
                fnT fn = subtract_inplace_row_matrix_broadcast_impl<T1, T2>;
                return fn;
            }
        }
    }
};

} // namespace subtract
} // namespace kernels
} // namespace tensor
} // namespace dpctl
