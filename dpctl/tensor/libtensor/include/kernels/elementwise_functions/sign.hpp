//=== sign.hpp -   Unary function SIGN                   ------  *-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of SIGN(x) function.
//===---------------------------------------------------------------------===//

#pragma once
#include <CL/sycl.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
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
namespace sign
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;
using dpctl::tensor::type_utils::vec_cast;

template <typename argT, typename resT> struct SignFunctor
{
    static_assert(std::is_same_v<resT, argT>);
    using is_constant = typename std::false_type;
    // constexpr resT constant_value = resT{};
    using supports_vec = typename std::negation<
        std::disjunction<is_complex<resT>, is_complex<argT>>>;
    using supports_sg_loadstore = std::false_type;

    resT operator()(const argT &x) const
    {
        if constexpr (std::is_integral_v<argT>) {
            if constexpr (std::is_unsigned_v<argT>) {
                return resT(0 < x);
            }
            else {
                return sign<argT>(x);
            }
        }
        else {
            if constexpr (is_complex<argT>::value) {
                if (x == argT(0)) {
                    return resT(0);
                }
                else {
                    return (x / std::abs(x));
                }
            }
            else {
                if (std::isnan(x)) {
                    return std::numeric_limits<resT>::quiet_NaN();
                }
                else {
                    return sign<argT>(x);
                }
            }
        }
    }

private:
    template <typename T> T sign(const T &v) const
    {
        return (T(0) < v) - (v < T(0));
    }
};

template <typename argT,
          typename resT = argT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2>
using SignContigFunctor = elementwise_common::
    UnaryContigFunctor<argT, resT, SignFunctor<argT, resT>, vec_sz, n_vecs>;

template <typename T> struct SignOutputType
{
    using value_type = typename std::disjunction< // disjunction is C++17
                                                  // feature, supported by DPC++
        td_ns::TypeMapResultEntry<T, std::uint8_t>,
        td_ns::TypeMapResultEntry<T, std::uint16_t>,
        td_ns::TypeMapResultEntry<T, std::uint32_t>,
        td_ns::TypeMapResultEntry<T, std::uint64_t>,
        td_ns::TypeMapResultEntry<T, std::int8_t>,
        td_ns::TypeMapResultEntry<T, std::int16_t>,
        td_ns::TypeMapResultEntry<T, std::int32_t>,
        td_ns::TypeMapResultEntry<T, std::int64_t>,
        td_ns::TypeMapResultEntry<T, sycl::half>,
        td_ns::TypeMapResultEntry<T, float>,
        td_ns::TypeMapResultEntry<T, double>,
        td_ns::TypeMapResultEntry<T, std::complex<float>>,
        td_ns::TypeMapResultEntry<T, std::complex<double>>,
        td_ns::DefaultResultEntry<void>>::result_type;
};

template <typename T1, typename T2, unsigned int vec_sz, unsigned int n_vecs>
class sign_contig_kernel;

template <typename argTy>
sycl::event sign_contig_impl(sycl::queue &exec_q,
                             size_t nelems,
                             const char *arg_p,
                             char *res_p,
                             const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::unary_contig_impl<
        argTy, SignOutputType, SignContigFunctor, sign_contig_kernel>(
        exec_q, nelems, arg_p, res_p, depends);
}

template <typename fnT, typename T> struct SignContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename SignOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = sign_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T> struct SignTypeMapFactory
{
    /*! @brief get typeid for output type of sign(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename SignOutputType<T>::value_type;
        ;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename argTy, typename resTy, typename IndexerT>
using SignStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, SignFunctor<argTy, resTy>>;

template <typename T1, typename T2, typename T3> class sign_strided_kernel;

template <typename argTy>
sycl::event
sign_strided_impl(sycl::queue &exec_q,
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
        argTy, SignOutputType, SignStridedFunctor, sign_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T> struct SignStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename SignOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = sign_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace sign
} // namespace kernels
} // namespace tensor
} // namespace dpctl
