//=== tan.hpp -   Unary function TAN                    ------  *-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of TAN(x) function.
//===---------------------------------------------------------------------===//

#pragma once
#include <cmath>
#include <complex>
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
namespace tan
{

namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;

template <typename argT, typename resT> struct TanFunctor
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
            /*
             * since tan(in) = -I * tanh(I * in), for special cases,
             * we calculate real and imaginary parts of z = tanh(I * in) and
             * return { imag(z) , -real(z) } which is tan(in).
             */
            const realT x = -std::imag(in);
            const realT y = std::real(in);
            /*
             * tanh(NaN + i 0) = NaN + i 0
             *
             * tanh(NaN + i y) = NaN + i NaN        for y != 0
             *
             * The imaginary part has the sign of x*sin(2*y), but there's no
             * special effort to get this right.
             *
             * tanh(+-Inf +- i Inf) = +-1 +- 0
             *
             * tanh(+-Inf + i y) = +-1 + 0 sin(2y)        for y finite
             *
             * The imaginary part of the sign is unspecified.  This special
             * case is only needed to avoid a spurious invalid exception when
             * y is infinite.
             */
            if (!std::isfinite(x)) {
                if (std::isnan(x)) {
                    const realT tanh_re = x;
                    const realT tanh_im = (y == realT(0) ? y : x * y);
                    return resT{tanh_im, -tanh_re};
                }
                const realT tanh_re = sycl::copysign(realT(1), x);
                const realT tanh_im = sycl::copysign(
                    realT(0), std::isinf(y) ? y : std::sin(y) * std::cos(y));
                return resT{tanh_im, -tanh_re};
            }
            /*
             * tanh(x + i NAN) = NaN + i NaN for non-zero x
             * tanh(x +- i Inf) = NaN + i NaN for non-zero x
             * tanh(0 + i NAN) = 0 + i NaN
             * tanh(0 +- i Inf) = 0 + i NaN
             */
            if (!std::isfinite(y)) {
                if (x == realT(0)) {
                    return resT{q_nan, x};
                }
                return resT{q_nan, q_nan};
            }
            /* ordinary cases */
#ifdef USE_SYCL_FOR_COMPLEX_TYPES
            return exprm_ns::tan(exprm_ns::complex<realT>(in)); // std::tan(in);
#else
            return std::tan(in);
#endif
        }
        else {
            static_assert(std::is_floating_point_v<argT> ||
                          std::is_same_v<argT, sycl::half>);
            return std::tan(in);
        }
    }
};

template <typename argTy,
          typename resTy = argTy,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using TanContigFunctor =
    elementwise_common::UnaryContigFunctor<argTy,
                                           resTy,
                                           TanFunctor<argTy, resTy>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename argTy, typename resTy, typename IndexerT>
using TanStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, TanFunctor<argTy, resTy>>;

template <typename T> struct TanOutputType
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
class tan_contig_kernel;

template <typename argTy>
sycl::event tan_contig_impl(sycl::queue &exec_q,
                            size_t nelems,
                            const char *arg_p,
                            char *res_p,
                            const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::unary_contig_impl<
        argTy, TanOutputType, TanContigFunctor, tan_contig_kernel>(
        exec_q, nelems, arg_p, res_p, depends);
}

template <typename fnT, typename T> struct TanContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename TanOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = tan_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T> struct TanTypeMapFactory
{
    /*! @brief get typeid for output type of std::tan(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename TanOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3> class tan_strided_kernel;

template <typename argTy>
sycl::event tan_strided_impl(sycl::queue &exec_q,
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
        argTy, TanOutputType, TanStridedFunctor, tan_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T> struct TanStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename TanOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = tan_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace tan
} // namespace kernels
} // namespace tensor
} // namespace dpctl
