//=== square.hpp -   Unary function SQUARE       ------         *-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of SQUARE(x)
///
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
namespace square
{

namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;
using dpctl::tensor::type_utils::vec_cast;

template <typename argT, typename resT> struct SquareFunctor
{

    // is function constant for given argT
    using is_constant = typename std::false_type;
    // constant value, if constant
    // constexpr resT constant_value = resT{};
    // is function defined for sycl::vec
    using supports_vec = typename std::negation<
        std::disjunction<is_complex<resT>, is_complex<argT>>>;
    // do both argTy and resTy support sugroup store/load operation
    using supports_sg_loadstore = typename std::negation<
        std::disjunction<is_complex<resT>, is_complex<argT>>>;

    resT operator()(const argT &in) const
    {
        if constexpr (is_complex<argT>::value) {
#ifdef USE_SYCL_FOR_COMPLEX_TYPES
            using realT = typename argT::value_type;

            auto z = exprm_ns::complex<realT>(in);

            return z * z;
#else
            return in * in;
#endif
        }
        else {
            return in * in;
        }
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz> operator()(const sycl::vec<argT, vec_sz> &in) const
    {
        auto const &res_vec = in * in;
        using deducedT = typename std::remove_cv_t<
            std::remove_reference_t<decltype(res_vec)>>::element_type;
        if constexpr (std::is_same_v<resT, deducedT>) {
            return res_vec;
        }
        else {
            return vec_cast<resT, deducedT, vec_sz>(res_vec);
        }
    }
};

template <typename argTy,
          typename resTy = argTy,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using SquareContigFunctor =
    elementwise_common::UnaryContigFunctor<argTy,
                                           resTy,
                                           SquareFunctor<argTy, resTy>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename argTy, typename resTy, typename IndexerT>
using SquareStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, SquareFunctor<argTy, resTy>>;

template <typename T> struct SquareOutputType
{
    using value_type = typename std::disjunction< // disjunction is C++17
                                                  // feature, supported by DPC++
        td_ns::TypeMapResultEntry<T, bool, std::int8_t>,
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
class square_contig_kernel;

template <typename argTy>
sycl::event square_contig_impl(sycl::queue &exec_q,
                               size_t nelems,
                               const char *arg_p,
                               char *res_p,
                               const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::unary_contig_impl<
        argTy, SquareOutputType, SquareContigFunctor, square_contig_kernel>(
        exec_q, nelems, arg_p, res_p, depends);
}

template <typename fnT, typename T> struct SquareContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename SquareOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = square_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T> struct SquareTypeMapFactory
{
    /*! @brief get typeid for output type of x * x */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename SquareOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3> class square_strided_kernel;

template <typename argTy>
sycl::event
square_strided_impl(sycl::queue &exec_q,
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
        argTy, SquareOutputType, SquareStridedFunctor, square_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T> struct SquareStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename SquareOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = square_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace square
} // namespace kernels
} // namespace tensor
} // namespace dpctl
