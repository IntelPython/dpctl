//=== cbrt.hpp -   Unary function CBRT                   ------  *-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of CBRT(x)
/// function that compute a square root.
//===---------------------------------------------------------------------===//

#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "vec_size_util.hpp"

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
namespace cbrt
{

using dpctl::tensor::ssize_t;
namespace td_ns = dpctl::tensor::type_dispatch;

template <typename argT, typename resT> struct CbrtFunctor
{

    // is function constant for given argT
    using is_constant = typename std::false_type;
    // constant value, if constant
    // constexpr resT constant_value = resT{};
    // is function defined for sycl::vec
    using supports_vec = typename std::false_type;
    // do both argTy and resTy support sugroup store/load operation
    using supports_sg_loadstore = typename std::true_type;

    resT operator()(const argT &in) const { return sycl::cbrt(in); }
};

template <typename argTy,
          typename resTy = argTy,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
using CbrtContigFunctor =
    elementwise_common::UnaryContigFunctor<argTy,
                                           resTy,
                                           CbrtFunctor<argTy, resTy>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename argTy, typename resTy, typename IndexerT>
using CbrtStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, CbrtFunctor<argTy, resTy>>;

template <typename T> struct CbrtOutputType
{
    using value_type = typename std::disjunction<
        td_ns::TypeMapResultEntry<T, sycl::half, sycl::half>,
        td_ns::TypeMapResultEntry<T, float, float>,
        td_ns::TypeMapResultEntry<T, double, double>,
        td_ns::DefaultResultEntry<void>>::result_type;

    static constexpr bool is_defined = !std::is_same_v<value_type, void>;
};

namespace hyperparam_detail
{

namespace vsu_ns = dpctl::tensor::kernels::vec_size_utils;

using vsu_ns::ContigHyperparameterSetDefault;
using vsu_ns::UnaryContigHyperparameterSetEntry;

template <typename argTy> struct CbrtContigHyperparameterSet
{
    using value_type =
        typename std::disjunction<ContigHyperparameterSetDefault<4u, 2u>>;

    constexpr static auto vec_sz = value_type::vec_sz;
    constexpr static auto n_vecs = value_type::n_vecs;
};

} // end of namespace hyperparam_detail

template <typename T1, typename T2, std::uint8_t vec_sz, std::uint8_t n_vecs>
class cbrt_contig_kernel;

template <typename argTy>
sycl::event cbrt_contig_impl(sycl::queue &exec_q,
                             std::size_t nelems,
                             const char *arg_p,
                             char *res_p,
                             const std::vector<sycl::event> &depends = {})
{
    using CbrtHS = hyperparam_detail::CbrtContigHyperparameterSet<argTy>;
    static constexpr std::uint8_t vec_sz = CbrtHS::vec_sz;
    static constexpr std::uint8_t n_vecs = CbrtHS::n_vecs;

    return elementwise_common::unary_contig_impl<
        argTy, CbrtOutputType, CbrtContigFunctor, cbrt_contig_kernel, vec_sz,
        n_vecs>(exec_q, nelems, arg_p, res_p, depends);
}

template <typename fnT, typename T> struct CbrtContigFactory
{
    fnT get()
    {
        if constexpr (!CbrtOutputType<T>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = cbrt_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T> struct CbrtTypeMapFactory
{
    /*! @brief get typeid for output type of std::cbrt(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename CbrtOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3> class cbrt_strided_kernel;

template <typename argTy>
sycl::event
cbrt_strided_impl(sycl::queue &exec_q,
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
    return elementwise_common::unary_strided_impl<
        argTy, CbrtOutputType, CbrtStridedFunctor, cbrt_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T> struct CbrtStridedFactory
{
    fnT get()
    {
        if constexpr (!CbrtOutputType<T>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = cbrt_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace cbrt
} // namespace kernels
} // namespace tensor
} // namespace dpctl
