//=== cos.hpp -   Unary function COS                     ------  *-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of COS(x) function.
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
namespace cos
{

namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;

template <typename argT, typename resT> struct CosFunctor
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

            realT const &in_re = std::real(in);
            realT const &in_im = std::imag(in);

            const bool in_re_finite = std::isfinite(in_re);
            const bool in_im_finite = std::isfinite(in_im);

            /*
             * Handle the nearly-non-exceptional cases where
             * real and imaginary parts of input are finite.
             */
            if (in_re_finite && in_im_finite) {
#ifdef USE_SYCL_FOR_COMPLEX_TYPES
                return exprm_ns::cos(
                    exprm_ns::complex<realT>(in)); // std::cos(in);
#else
                return std::cos(in);
#endif
            }

            /*
             * since cos(in) = cosh(I * in), for special cases,
             * we return cosh(I * in).
             */
            const realT x = -in_im;
            const realT y = in_re;

            const bool xfinite = in_im_finite;
            const bool yfinite = in_re_finite;
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
                const realT y_m_y = (y - y);
                const realT res_im = sycl::copysign(realT(0), x * y_m_y);
                return resT{y_m_y, res_im};
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
                const realT y_m_y = (y - y);
                return resT{y_m_y, x * y_m_y};
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
                    return resT{x * x, sycl::copysign(q_nan, x)};
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
            return resT{(x * x) * q_nan, (x + x) * q_nan};
        }
        else {
            static_assert(std::is_floating_point_v<argT> ||
                          std::is_same_v<argT, sycl::half>);
            return std::cos(in);
        }
    }
};

template <typename argTy,
          typename resTy = argTy,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using CosContigFunctor =
    elementwise_common::UnaryContigFunctor<argTy,
                                           resTy,
                                           CosFunctor<argTy, resTy>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename argTy, typename resTy, typename IndexerT>
using CosStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, CosFunctor<argTy, resTy>>;

template <typename T> struct CosOutputType
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
class cos_contig_kernel;

template <typename argTy>
sycl::event cos_contig_impl(sycl::queue &exec_q,
                            size_t nelems,
                            const char *arg_p,
                            char *res_p,
                            const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::unary_contig_impl<
        argTy, CosOutputType, CosContigFunctor, cos_contig_kernel>(
        exec_q, nelems, arg_p, res_p, depends);
}

template <typename fnT, typename T> struct CosContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename CosOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = cos_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T> struct CosTypeMapFactory
{
    /*! @brief get typeid for output type of sycl::cos(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename CosOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3> class cos_strided_kernel;

template <typename argTy>
sycl::event cos_strided_impl(sycl::queue &exec_q,
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
        argTy, CosOutputType, CosStridedFunctor, cos_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T> struct CosStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename CosOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = cos_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace cos
} // namespace kernels
} // namespace tensor
} // namespace dpctl
