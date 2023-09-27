//=== asinh.hpp -   Unary function ASINH                ------  *-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of ASINH(x) function.
//===---------------------------------------------------------------------===//

#pragma once
#include <CL/sycl.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "kernels/elementwise_functions/common.hpp"

#include "utils/offset_utils.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"
#include <pybind11/pybind11.h>

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace asinh
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;

template <typename argT, typename resT> struct AsinhFunctor
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

            if (std::isnan(x)) {
                /* asinh(NaN + I*+-Inf) = opt(+-)Inf + I*NaN */
                if (std::isinf(y)) {
                    return resT{y, q_nan};
                }
                /* asinh(NaN + I*0) = NaN + I*0 */
                if (y == realT(0)) {
                    return resT{q_nan, y};
                }
                /* All other cases involving NaN return NaN + I*NaN. */
                return resT{q_nan, q_nan};
            }

            if (std::isnan(y)) {
                /* asinh(+-Inf + I*NaN) = +-Inf + I*NaN */
                if (std::isinf(x)) {
                    return resT{x, q_nan};
                }
                /* All other cases involving NaN return NaN + I*NaN. */
                return resT{q_nan, q_nan};
            }

            /*
             * For large x or y including asinh(+-Inf + I*+-Inf)
             * asinh(in) = sign(x)*log(sign(x)*in) + O(1/in^2)   as in ->
             * infinity The above formula works for the imaginary part as well,
             * because Im(asinh(in)) = sign(x)*atan2(sign(x)*y, fabs(x)) +
             * O(y/in^3) as in -> infinity, uniformly in y
             */
            constexpr realT r_eps =
                realT(1) / std::numeric_limits<realT>::epsilon();

            if (std::abs(x) > r_eps || std::abs(y) > r_eps) {
                resT log_in = (std::signbit(x)) ? std::log(-in) : std::log(in);
                realT wx = std::real(log_in) + std::log(realT(2));
                realT wy = std::imag(log_in);
                const realT res_re = std::copysign(wx, x);
                const realT res_im = std::copysign(wy, y);
                return resT{res_re, res_im};
            }

            /* ordinary cases */
            return std::asinh(in);
        }
        else {
            static_assert(std::is_floating_point_v<argT> ||
                          std::is_same_v<argT, sycl::half>);
            return std::asinh(in);
        }
    }
};

template <typename argTy,
          typename resTy = argTy,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2>
using AsinhContigFunctor =
    elementwise_common::UnaryContigFunctor<argTy,
                                           resTy,
                                           AsinhFunctor<argTy, resTy>,
                                           vec_sz,
                                           n_vecs>;

template <typename argTy, typename resTy, typename IndexerT>
using AsinhStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, AsinhFunctor<argTy, resTy>>;

template <typename T> struct AsinhOutputType
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
class asinh_contig_kernel;

template <typename argTy>
sycl::event asinh_contig_impl(sycl::queue &exec_q,
                              size_t nelems,
                              const char *arg_p,
                              char *res_p,
                              const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::unary_contig_impl<
        argTy, AsinhOutputType, AsinhContigFunctor, asinh_contig_kernel>(
        exec_q, nelems, arg_p, res_p, depends);
}

template <typename fnT, typename T> struct AsinhContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename AsinhOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = asinh_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T> struct AsinhTypeMapFactory
{
    /*! @brief get typeid for output type of std::asinh(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename AsinhOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3> class asinh_strided_kernel;

template <typename argTy>
sycl::event
asinh_strided_impl(sycl::queue &exec_q,
                   size_t nelems,
                   int nd,
                   const py::ssize_t *shape_and_strides,
                   const char *arg_p,
                   py::ssize_t arg_offset,
                   char *res_p,
                   py::ssize_t res_offset,
                   const std::vector<sycl::event> &depends,
                   const std::vector<sycl::event> &additional_depends)
{
    return elementwise_common::unary_strided_impl<
        argTy, AsinhOutputType, AsinhStridedFunctor, asinh_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T> struct AsinhStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename AsinhOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = asinh_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace asinh
} // namespace kernels
} // namespace tensor
} // namespace dpctl
