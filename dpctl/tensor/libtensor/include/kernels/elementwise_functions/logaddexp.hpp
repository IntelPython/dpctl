//=== logaddexp.hpp -   Binary function LOGADDEXP                    ------
//*-C++-*--/===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2023 Intel Corporation
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
/// This file defines kernels for elementwise evaluation of LOGADDEXP(x1, x2)
/// function.
//===---------------------------------------------------------------------===//

#pragma once
#include <CL/sycl.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "utils/offset_utils.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

#include "kernels/elementwise_functions/common.hpp"
#include <pybind11/pybind11.h>

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace logaddexp
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;
namespace tu_ns = dpctl::tensor::type_utils;

using dpctl::tensor::type_utils::is_complex;
using dpctl::tensor::type_utils::vec_cast;

template <typename argT1, typename argT2, typename resT> struct LogAddExpFunctor
{
    using supports_sg_loadstore = std::true_type;
    using supports_vec = std::true_type;

    resT operator()(const argT1 &in1, const argT2 &in2) const
    {
        return impl<resT>(in1, in2);
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz>
    operator()(const sycl::vec<argT1, vec_sz> &in1,
               const sycl::vec<argT2, vec_sz> &in2) const
    {
        sycl::vec<resT, vec_sz> res;
        auto diff = in1 - in2; // take advantange of faster vec arithmetic

#pragma unroll
        for (int i = 0; i < vec_sz; ++i) {
            if (std::isfinite(diff[i])) {
                res[i] = std::max<resT>(in1[i], in2[i]) +
                         impl_finite<resT>(-std::abs(diff[i]));
            }
            else {
                res[i] = impl<resT>(in1[i], in2[i]);
            }
        }

        return res;
    }

private:
    template <typename T> T impl(T const &in1, T const &in2) const
    {
        if (in1 == in2) { // handle signed infinities
            const T log2 = std::log(T(2));
            return in1 + log2;
        }
        else {
            const T tmp = in1 - in2;
            if (tmp > 0) {
                return in1 + std::log1p(std::exp(-tmp));
            }
            else if (tmp <= 0) {
                return in2 + std::log1p(std::exp(tmp));
            }
            else {
                return std::numeric_limits<T>::quiet_NaN();
            }
        }
    }

    template <typename T> T impl_finite(T const &in) const
    {
        return (in > 0) ? (in + std::log1p(std::exp(-in)))
                        : std::log1p(std::exp(in));
    }
};

template <typename argT1,
          typename argT2,
          typename resT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2>
using LogAddExpContigFunctor = elementwise_common::BinaryContigFunctor<
    argT1,
    argT2,
    resT,
    LogAddExpFunctor<argT1, argT2, resT>,
    vec_sz,
    n_vecs>;

template <typename argT1, typename argT2, typename resT, typename IndexerT>
using LogAddExpStridedFunctor = elementwise_common::BinaryStridedFunctor<
    argT1,
    argT2,
    resT,
    IndexerT,
    LogAddExpFunctor<argT1, argT2, resT>>;

template <typename T1, typename T2> struct LogAddExpOutputType
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
class logaddexp_contig_kernel;

template <typename argTy1, typename argTy2>
sycl::event logaddexp_contig_impl(sycl::queue &exec_q,
                                  size_t nelems,
                                  const char *arg1_p,
                                  py::ssize_t arg1_offset,
                                  const char *arg2_p,
                                  py::ssize_t arg2_offset,
                                  char *res_p,
                                  py::ssize_t res_offset,
                                  const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::binary_contig_impl<
        argTy1, argTy2, LogAddExpOutputType, LogAddExpContigFunctor,
        logaddexp_contig_kernel>(exec_q, nelems, arg1_p, arg1_offset, arg2_p,
                                 arg2_offset, res_p, res_offset, depends);
}

template <typename fnT, typename T1, typename T2> struct LogAddExpContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename LogAddExpOutputType<T1, T2>::value_type,
                          void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = logaddexp_contig_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2> struct LogAddExpTypeMapFactory
{
    /*! @brief get typeid for output type of std::logaddexp(T1 x, T2 y) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename LogAddExpOutputType<T1, T2>::value_type;
        ;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename resT, typename IndexerT>
class logaddexp_strided_kernel;

template <typename argTy1, typename argTy2>
sycl::event
logaddexp_strided_impl(sycl::queue &exec_q,
                       size_t nelems,
                       int nd,
                       const py::ssize_t *shape_and_strides,
                       const char *arg1_p,
                       py::ssize_t arg1_offset,
                       const char *arg2_p,
                       py::ssize_t arg2_offset,
                       char *res_p,
                       py::ssize_t res_offset,
                       const std::vector<sycl::event> &depends,
                       const std::vector<sycl::event> &additional_depends)
{
    return elementwise_common::binary_strided_impl<
        argTy1, argTy2, LogAddExpOutputType, LogAddExpStridedFunctor,
        logaddexp_strided_kernel>(exec_q, nelems, nd, shape_and_strides, arg1_p,
                                  arg1_offset, arg2_p, arg2_offset, res_p,
                                  res_offset, depends, additional_depends);
}

template <typename fnT, typename T1, typename T2> struct LogAddExpStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename LogAddExpOutputType<T1, T2>::value_type,
                          void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = logaddexp_strided_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename argT1, typename argT2, typename resT>
class logaddexp_matrix_row_broadcast_sg_krn;

} // namespace logaddexp
} // namespace kernels
} // namespace tensor
} // namespace dpctl
