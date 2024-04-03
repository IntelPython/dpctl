//=== log1p.hpp -   Unary function LOG1P                   ------
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
/// This file defines kernels for elementwise evaluation of LOG1P(x) function.
//===---------------------------------------------------------------------===//

#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
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
namespace log1p
{

namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;

// TODO: evaluate precision against alternatives
template <typename argT, typename resT> struct Log1pFunctor
{

    // is function constant for given argT
    using is_constant = typename std::false_type;
    // constant value, if constant
    // constexpr resT constant_value = resT{};
    // is function defined for sycl::vec
    using supports_vec = typename std::false_type;
    // do both argTy and resTy support sugroup store/load operation
    using supports_sg_loadstore = typename std::negation<
        std::disjunction<is_complex<resT>, is_complex<argT>>>;

    resT operator()(const argT &in) const
    {
        if constexpr (is_complex<argT>::value) {
            // log1p(z) = ln((x + 1) + yI)
            //          = ln(|(x + 1) + yi|)
            //             + I * atan2(y, x + 1)
            //          = ln(sqrt((x + 1)^2 + y^2))
            //             + I *atan2(y, x + 1)
            //          = log1p(x^2 + 2x + y^2) / 2
            //             + I * atan2(y, x + 1)
            using realT = typename argT::value_type;
            const realT x = std::real(in);
            const realT y = std::imag(in);

            // imaginary part of result
            const realT res_im = std::atan2(y, x + 1);

            if (std::max(sycl::fabs(x), sycl::fabs(y)) < realT{.1}) {
                const realT v = x * (2 + x) + y * y;
                return resT{std::log1p(v) / 2, res_im};
            }
            else {
                // when not close to zero,
                // prevent overflow
                const realT m = std::hypot(x + 1, y);
                return resT{std::log(m), res_im};
            }
        }
        else {
            static_assert(std::is_floating_point_v<argT> ||
                          std::is_same_v<argT, sycl::half>);
            return std::log1p(in);
        }
    }
};

template <typename argTy,
          typename resTy = argTy,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using Log1pContigFunctor =
    elementwise_common::UnaryContigFunctor<argTy,
                                           resTy,
                                           Log1pFunctor<argTy, resTy>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename argTy, typename resTy, typename IndexerT>
using Log1pStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, Log1pFunctor<argTy, resTy>>;

template <typename T> struct Log1pOutputType
{
    using value_type = typename std::disjunction< // disjunction is C++17
                                                  // feature, supported by DPC++
        td_ns::TypeMapResultEntry<T, sycl::half, sycl::half>,
        td_ns::TypeMapResultEntry<T, float, float>,
        td_ns::TypeMapResultEntry<T, double, double>,
        td_ns::TypeMapResultEntry<T, std::complex<float>, std::complex<float>>,
        td_ns::
            TypeMapResultEntry<T, std::complex<double>, std::complex<double>>,
        td_ns::DefaultResultEntry<void>>::result_type;
};

template <typename T1, typename T2, unsigned int vec_sz, unsigned int n_vecs>
class log1p_contig_kernel;

template <typename argTy>
sycl::event log1p_contig_impl(sycl::queue &exec_q,
                              size_t nelems,
                              const char *arg_p,
                              char *res_p,
                              const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::unary_contig_impl<
        argTy, Log1pOutputType, Log1pContigFunctor, log1p_contig_kernel>(
        exec_q, nelems, arg_p, res_p, depends);
}

template <typename fnT, typename T> struct Log1pContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename Log1pOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = log1p_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T> struct Log1pTypeMapFactory
{
    /*! @brief get typeid for output type of std::log1p(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename Log1pOutputType<T>::value_type;
        ;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3> class log1p_strided_kernel;

template <typename argTy>
sycl::event
log1p_strided_impl(sycl::queue &exec_q,
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
        argTy, Log1pOutputType, Log1pStridedFunctor, log1p_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T> struct Log1pStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename Log1pOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = log1p_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace log1p
} // namespace kernels
} // namespace tensor
} // namespace dpctl
