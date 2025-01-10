//=== isfinite.hpp -   Unary function ISFINITE           ------  *-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of ISFINITE(x)
/// function that tests whether a tensor element is finite.
//===---------------------------------------------------------------------===//

#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "vec_size_util.hpp"

#include "utils/offset_utils.hpp"
#include "utils/type_dispatch_building.hpp"
#include "utils/type_utils.hpp"

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace isfinite
{

using dpctl::tensor::ssize_t;
namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;
using dpctl::tensor::type_utils::vec_cast;

template <typename argT, typename resT> struct IsFiniteFunctor
{
    static_assert(std::is_same_v<resT, bool>);

    /*
    std::is_same<argT, bool>::value ||
                           std::is_integral<argT>::value
    */
    using is_constant = typename std::disjunction<std::is_same<argT, bool>,
                                                  std::is_integral<argT>>;
    static constexpr resT constant_value = true;
    using supports_vec = typename std::false_type;
    using supports_sg_loadstore = typename std::negation<
        std::disjunction<is_complex<resT>, is_complex<argT>>>;

    resT operator()(const argT &in) const
    {
        if constexpr (is_complex<argT>::value) {
            const bool real_isfinite = std::isfinite(std::real(in));
            const bool imag_isfinite = std::isfinite(std::imag(in));
            return (real_isfinite && imag_isfinite);
        }
        else if constexpr (std::is_same<argT, bool>::value ||
                           std::is_integral<argT>::value)
        {
            return constant_value;
        }
        else if constexpr (std::is_same_v<argT, sycl::half>) {
            return sycl::isfinite(in);
        }
        else {
            return std::isfinite(in);
        }
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz> operator()(const sycl::vec<argT, vec_sz> &in) const
    {
        auto const &res_vec = sycl::isfinite(in);

        using deducedT = typename std::remove_cv_t<
            std::remove_reference_t<decltype(res_vec)>>::element_type;

        return vec_cast<bool, deducedT, vec_sz>(res_vec);
    }
};

template <typename argT,
          typename resT = bool,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
using IsFiniteContigFunctor =
    elementwise_common::UnaryContigFunctor<argT,
                                           resT,
                                           IsFiniteFunctor<argT, resT>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename argTy, typename resTy, typename IndexerT>
using IsFiniteStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, IsFiniteFunctor<argTy, resTy>>;

template <typename argTy> struct IsFiniteOutputType
{
    using value_type = bool;
};

namespace hyperparam_detail
{

namespace vsu_ns = dpctl::tensor::kernels::vec_size_utils;

using vsu_ns::ContigHyperparameterSetDefault;
using vsu_ns::UnaryContigHyperparameterSetEntry;

template <typename argTy> struct IsFiniteContigHyperparameterSet
{
    using value_type =
        typename std::disjunction<ContigHyperparameterSetDefault<4u, 2u>>;

    constexpr static auto vec_sz = value_type::vec_sz;
    constexpr static auto n_vecs = value_type::n_vecs;
};

} // end of namespace hyperparam_detail

template <typename T1, typename T2, std::uint8_t vec_sz, std::uint8_t n_vecs>
class isfinite_contig_kernel;

template <typename argTy>
sycl::event isfinite_contig_impl(sycl::queue &exec_q,
                                 std::size_t nelems,
                                 const char *arg_p,
                                 char *res_p,
                                 const std::vector<sycl::event> &depends = {})
{
    using IsFiniteHS =
        hyperparam_detail::IsFiniteContigHyperparameterSet<argTy>;
    constexpr std::uint8_t vec_sz = IsFiniteHS::vec_sz;
    constexpr std::uint8_t n_vecs = IsFiniteHS::n_vecs;

    return elementwise_common::unary_contig_impl<
        argTy, IsFiniteOutputType, IsFiniteContigFunctor,
        isfinite_contig_kernel, vec_sz, n_vecs>(exec_q, nelems, arg_p, res_p,
                                                depends);
}

template <typename fnT, typename T> struct IsFiniteContigFactory
{
    fnT get()
    {
        fnT fn = isfinite_contig_impl<T>;
        return fn;
    }
};

template <typename fnT, typename T> struct IsFiniteTypeMapFactory
{
    /*! @brief get typeid for output type of sycl::isfinite(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename IsFiniteOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3> class isfinite_strided_kernel;

template <typename argTy>
sycl::event
isfinite_strided_impl(sycl::queue &exec_q,
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
    return elementwise_common::unary_strided_impl<argTy, IsFiniteOutputType,
                                                  IsFiniteStridedFunctor,
                                                  isfinite_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T> struct IsFiniteStridedFactory
{
    fnT get()
    {
        fnT fn = isfinite_strided_impl<T>;
        return fn;
    }
};

} // namespace isfinite
} // namespace kernels
} // namespace tensor
} // namespace dpctl
