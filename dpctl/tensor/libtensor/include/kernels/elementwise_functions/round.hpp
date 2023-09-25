//=== round.hpp -   Unary function ROUND                ------  *-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of ROUND(x) function.
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
namespace round
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;

template <typename argT, typename resT> struct RoundFunctor
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

        if constexpr (std::is_integral_v<argT>) {
            return in;
        }
        else if constexpr (is_complex<argT>::value) {
            using realT = typename argT::value_type;
            return resT{round_func<realT>(std::real(in)),
                        round_func<realT>(std::imag(in))};
        }
        else {
            return round_func<argT>(in);
        }
    }

private:
    template <typename T> T round_func(const T &input) const
    {
        return std::rint(input);
    }
};

template <typename argTy,
          typename resTy = argTy,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2>
using RoundContigFunctor =
    elementwise_common::UnaryContigFunctor<argTy,
                                           resTy,
                                           RoundFunctor<argTy, resTy>,
                                           vec_sz,
                                           n_vecs>;

template <typename argTy, typename resTy, typename IndexerT>
using RoundStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, RoundFunctor<argTy, resTy>>;

template <typename T> struct RoundOutputType
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
class round_contig_kernel;

template <typename argTy>
sycl::event round_contig_impl(sycl::queue &exec_q,
                              size_t nelems,
                              const char *arg_p,
                              char *res_p,
                              const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::unary_contig_impl<
        argTy, RoundOutputType, RoundContigFunctor, round_contig_kernel>(
        exec_q, nelems, arg_p, res_p, depends);
}

template <typename fnT, typename T> struct RoundContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename RoundOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = round_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T> struct RoundTypeMapFactory
{
    /*! @brief get typeid for output type of sycl::round(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename RoundOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3> class round_strided_kernel;

template <typename argTy>
sycl::event
round_strided_impl(sycl::queue &exec_q,
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
        argTy, RoundOutputType, RoundStridedFunctor, round_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T> struct RoundStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename RoundOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = round_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace round
} // namespace kernels
} // namespace tensor
} // namespace dpctl
