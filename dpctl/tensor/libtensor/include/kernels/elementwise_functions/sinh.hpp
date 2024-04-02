//=== sinh.hpp -   Unary function SINH                  ------  *-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of SINH(x) function.
//===---------------------------------------------------------------------===//

#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "kernels/elementwise_functions/common.hpp"
#include "sycl_complex.hpp"

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
namespace sinh
{

namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;

template <typename argT, typename resT> struct SinhFunctor
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
            using realT = typename argT::value_type;

            const realT x = std::real(in);
            const realT y = std::imag(in);

            const bool xfinite = std::isfinite(x);
            const bool yfinite = std::isfinite(y);

            /*
             * Handle the nearly-non-exceptional cases where
             * real and imaginary parts of input are finite.
             */
            if (xfinite && yfinite) {
#ifdef USE_SYCL_FOR_COMPLEX_TYPES
                return exprm_ns::sinh(exprm_ns::complex<realT>(in));
#else
                return std::sinh(in);
#endif
            }
            /*
             * sinh(+-0 +- I Inf) = sign(d(+-0, dNaN))0 + I dNaN.
             * The sign of 0 in the result is unspecified.  Choice = normally
             * the same as dNaN.
             *
             * sinh(+-0 +- I NaN) = sign(d(+-0, NaN))0 + I d(NaN).
             * The sign of 0 in the result is unspecified.  Choice = normally
             * the same as d(NaN).
             */
            if (x == realT(0) && !yfinite) {
                const realT res_re = sycl::copysign(realT(0), x * (y - y));
                return resT{res_re, y - y};
            }

            /*
             * sinh(+-Inf +- I 0) = +-Inf + I +-0.
             *
             * sinh(NaN +- I 0)   = d(NaN) + I +-0.
             */
            if (y == realT(0) && !xfinite) {
                if (std::isnan(x)) {
                    return resT{x, y};
                }
                const realT res_im = sycl::copysign(realT(0), y);
                return resT{x, res_im};
            }

            /*
             * sinh(x +- I Inf) = dNaN + I dNaN.
             *
             * sinh(x + I NaN) = d(NaN) + I d(NaN).
             */
            if (xfinite && !yfinite) {
                return resT{y - y, x * (y - y)};
            }

            /*
             * sinh(+-Inf + I NaN)  = +-Inf + I d(NaN).
             * The sign of Inf in the result is unspecified.  Choice = normally
             * the same as d(NaN).
             *
             * sinh(+-Inf +- I Inf) = +Inf + I dNaN.
             * The sign of Inf in the result is unspecified.  Choice = always +.
             *
             * sinh(+-Inf + I y)   = +-Inf cos(y) + I Inf sin(y)
             */
            if (!xfinite && !std::isnan(x)) {
                if (!yfinite) {
                    return resT{x * x, x * (y - y)};
                }
                return resT{x * std::cos(y),
                            std::numeric_limits<realT>::infinity() *
                                std::sin(y)};
            }

            /*
             * sinh(NaN + I NaN)  = d(NaN) + I d(NaN).
             *
             * sinh(NaN +- I Inf) = d(NaN) + I d(NaN).
             *
             * sinh(NaN + I y)    = d(NaN) + I d(NaN).
             */
            return resT{(x * x) * (y - y), (x + x) * (y - y)};
        }
        else {
            static_assert(std::is_floating_point_v<argT> ||
                          std::is_same_v<argT, sycl::half>);
            return std::sinh(in);
        }
    }
};

template <typename argTy,
          typename resTy = argTy,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using SinhContigFunctor =
    elementwise_common::UnaryContigFunctor<argTy,
                                           resTy,
                                           SinhFunctor<argTy, resTy>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename argTy, typename resTy, typename IndexerT>
using SinhStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, SinhFunctor<argTy, resTy>>;

template <typename T> struct SinhOutputType
{
    using value_type = typename std::disjunction< // disjunction is C++17
                                                  // feature, supported by DPC++
        td_ns::TypeMapResultEntry<T, sycl::half>,
        td_ns::TypeMapResultEntry<T, float>,
        td_ns::TypeMapResultEntry<T, double>,
        td_ns::TypeMapResultEntry<T, std::complex<float>>,
        td_ns::TypeMapResultEntry<T, std::complex<double>>,
        td_ns::DefaultResultEntry<void>>::result_type;
};

template <typename T1, typename T2, unsigned int vec_sz, unsigned int n_vecs>
class sinh_contig_kernel;

template <typename argTy>
sycl::event sinh_contig_impl(sycl::queue &exec_q,
                             size_t nelems,
                             const char *arg_p,
                             char *res_p,
                             const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::unary_contig_impl<
        argTy, SinhOutputType, SinhContigFunctor, sinh_contig_kernel>(
        exec_q, nelems, arg_p, res_p, depends);
}

template <typename fnT, typename T> struct SinhContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename SinhOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = sinh_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T> struct SinhTypeMapFactory
{
    /*! @brief get typeid for output type of std::sinh(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename SinhOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3> class sinh_strided_kernel;

template <typename argTy>
sycl::event
sinh_strided_impl(sycl::queue &exec_q,
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
        argTy, SinhOutputType, SinhStridedFunctor, sinh_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T> struct SinhStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename SinhOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = sinh_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace sinh
} // namespace kernels
} // namespace tensor
} // namespace dpctl
