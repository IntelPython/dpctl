//=== cosh.hpp -   Unary function COSH                  ------  *-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of COSH(x) function.
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
namespace cosh
{

namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;

template <typename argT, typename resT> struct CoshFunctor
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

            constexpr realT q_nan = std::numeric_limits<realT>::quiet_NaN();

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
                return exprm_ns::cosh(
                    exprm_ns::complex<realT>(in)); // std::cosh(in);
#else
                return std::cosh(in);
#endif
            }

            /*
             * cosh(+-0 +- I Inf) = dNaN + I sign(d(+-0, dNaN))0.
             * The sign of 0 in the result is unspecified.  Choice = normally
             * the same as dNaN.
             *
             * cosh(+-0 +- I NaN) = d(NaN) + I sign(d(+-0, NaN))0.
             * The sign of 0 in the result is unspecified.  Choice = normally
             * the same as d(NaN).
             */
            if (x == realT(0) && !yfinite) {
                const realT res_im = sycl::copysign(realT(0), x * q_nan);
                return resT{q_nan, res_im};
            }

            /*
             * cosh(+-Inf +- I 0) = +Inf + I (+-)(+-)0.
             *
             * cosh(NaN +- I 0)   = d(NaN) + I sign(d(NaN, +-0))0.
             * The sign of 0 in the result is unspecified.
             */
            if (y == realT(0) && !xfinite) {
                const realT res_im = sycl::copysign(realT(0), x) * y;
                return resT{x * x, res_im};
            }

            /*
             * cosh(x +- I Inf) = dNaN + I dNaN.
             *
             * cosh(x + I NaN) = d(NaN) + I d(NaN).
             */
            if (xfinite && !yfinite) {
                return resT{q_nan, x * q_nan};
            }

            /*
             * cosh(+-Inf + I NaN)  = +Inf + I d(NaN).
             *
             * cosh(+-Inf +- I Inf) = +Inf + I dNaN.
             * The sign of Inf in the result is unspecified.  Choice = always +.
             *
             * cosh(+-Inf + I y)   = +Inf cos(y) +- I Inf sin(y)
             */
            if (std::isinf(x)) {
                if (!yfinite) {
                    return resT{x * x, x * q_nan};
                }
                return resT{(x * x) * std::cos(y), x * std::sin(y)};
            }

            /*
             * cosh(NaN + I NaN)  = d(NaN) + I d(NaN).
             *
             * cosh(NaN +- I Inf) = d(NaN) + I d(NaN).
             *
             * cosh(NaN + I y)    = d(NaN) + I d(NaN).
             */
            return resT{(x * x) * (y - y), (x + x) * (y - y)};
        }
        else {
            static_assert(std::is_floating_point_v<argT> ||
                          std::is_same_v<argT, sycl::half>);
            return std::cosh(in);
        }
    }
};

template <typename argTy,
          typename resTy = argTy,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using CoshContigFunctor =
    elementwise_common::UnaryContigFunctor<argTy,
                                           resTy,
                                           CoshFunctor<argTy, resTy>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename argTy, typename resTy, typename IndexerT>
using CoshStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, CoshFunctor<argTy, resTy>>;

template <typename T> struct CoshOutputType
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
class cosh_contig_kernel;

template <typename argTy>
sycl::event cosh_contig_impl(sycl::queue &exec_q,
                             size_t nelems,
                             const char *arg_p,
                             char *res_p,
                             const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::unary_contig_impl<
        argTy, CoshOutputType, CoshContigFunctor, cosh_contig_kernel>(
        exec_q, nelems, arg_p, res_p, depends);
}

template <typename fnT, typename T> struct CoshContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename CoshOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = cosh_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T> struct CoshTypeMapFactory
{
    /*! @brief get typeid for output type of std::cosh(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename CoshOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3> class cosh_strided_kernel;

template <typename argTy>
sycl::event
cosh_strided_impl(sycl::queue &exec_q,
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
        argTy, CoshOutputType, CoshStridedFunctor, cosh_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T> struct CoshStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename CoshOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = cosh_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace cosh
} // namespace kernels
} // namespace tensor
} // namespace dpctl
