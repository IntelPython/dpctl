//=== negative.hpp -   Unary function POSITIVE           ------  *-C++-*--/===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2025 Intel Corporation
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
/// This file defines kernels for elementwise evaluation of POSITIVE(x)
/// function that returns x.
//===---------------------------------------------------------------------===//

#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "vec_size_util.hpp"

#include "kernels/dpctl_tensor_types.hpp"
#include "kernels/elementwise_functions/common.hpp"

#include "utils/offset_utils.hpp"
#include "utils/type_dispatch_building.hpp"
#include "utils/type_utils.hpp"

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace negative
{

using dpctl::tensor::ssize_t;
namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;
using dpctl::tensor::type_utils::vec_cast;

template <typename argT, typename resT> struct NegativeFunctor
{

    using is_constant = typename std::false_type;
    // constexpr resT constant_value = resT{};
    using supports_vec = typename std::false_type;
    using supports_sg_loadstore = typename std::negation<
        std::disjunction<is_complex<resT>, is_complex<argT>>>;

    resT operator()(const argT &x) const { return -x; }
};

template <typename argT,
          typename resT = argT,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
using NegativeContigFunctor =
    elementwise_common::UnaryContigFunctor<argT,
                                           resT,
                                           NegativeFunctor<argT, resT>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename T> struct NegativeOutputType
{
    using value_type = typename std::disjunction<
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

    static constexpr bool is_defined = !std::is_same_v<value_type, void>;
};

namespace hyperparam_detail
{

namespace vsu_ns = dpctl::tensor::kernels::vec_size_utils;

using vsu_ns::ContigHyperparameterSetDefault;
using vsu_ns::UnaryContigHyperparameterSetEntry;

template <typename argTy> struct NegativeContigHyperparameterSet
{
    using value_type =
        typename std::disjunction<ContigHyperparameterSetDefault<4u, 2u>>;

    constexpr static auto vec_sz = value_type::vec_sz;
    constexpr static auto n_vecs = value_type::n_vecs;
};

} // end of namespace hyperparam_detail

template <typename T1, typename T2, std::uint8_t vec_sz, std::uint8_t n_vecs>
class negative_contig_kernel;

template <typename argTy>
sycl::event negative_contig_impl(sycl::queue &exec_q,
                                 std::size_t nelems,
                                 const char *arg_p,
                                 char *res_p,
                                 const std::vector<sycl::event> &depends = {})
{
    using NegHS = hyperparam_detail::NegativeContigHyperparameterSet<argTy>;
    static constexpr std::uint8_t vec_sz = NegHS::vec_sz;
    static constexpr std::uint8_t n_vecs = NegHS::n_vecs;

    return elementwise_common::unary_contig_impl<
        argTy, NegativeOutputType, NegativeContigFunctor,
        negative_contig_kernel, vec_sz, n_vecs>(exec_q, nelems, arg_p, res_p,
                                                depends);
}

template <typename fnT, typename T> struct NegativeContigFactory
{
    fnT get()
    {
        if constexpr (!NegativeOutputType<T>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = negative_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T> struct NegativeTypeMapFactory
{
    /*! @brief get typeid for output type of std::negative(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename NegativeOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename argTy, typename resTy, typename IndexerT>
using NegativeStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, NegativeFunctor<argTy, resTy>>;

template <typename T1, typename T2, typename T3> class negative_strided_kernel;

template <typename argTy>
sycl::event
negative_strided_impl(sycl::queue &exec_q,
                      std::size_t nelems,
                      int nd,
                      const ssize_t *shape_and_strides,
                      const char *arg_p,
                      ssize_t arg_offset,
                      char *res_p,
                      ssize_t res_offset,
                      const std::vector<sycl::event> &depends,
                      const std::vector<sycl::event> &additional_depends)
{
    return elementwise_common::unary_strided_impl<argTy, NegativeOutputType,
                                                  NegativeStridedFunctor,
                                                  negative_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T> struct NegativeStridedFactory
{
    fnT get()
    {
        if constexpr (!NegativeOutputType<T>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = negative_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace negative
} // namespace kernels
} // namespace tensor
} // namespace dpctl
