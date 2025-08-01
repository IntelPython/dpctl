//=== log2.hpp -   Unary function LOG2                     ------
//*-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of LOG2(x) function.
//===---------------------------------------------------------------------===//

#pragma once
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "sycl_complex.hpp"
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
namespace log2
{

using dpctl::tensor::ssize_t;
namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;
using dpctl::tensor::type_utils::vec_cast;

template <typename argT, typename resT> struct Log2Functor
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
            using realT = typename argT::value_type;

            // log(in) / log(realT{2});
            return exprm_ns::log(exprm_ns::complex<realT>(in)) /
                   sycl::log(realT{2});
        }
        else {
            return sycl::log2(in);
        }
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz> operator()(const sycl::vec<argT, vec_sz> &in) const
    {
        auto const &res_vec = sycl::log2(in);
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
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
using Log2ContigFunctor =
    elementwise_common::UnaryContigFunctor<argTy,
                                           resTy,
                                           Log2Functor<argTy, resTy>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename argTy, typename resTy, typename IndexerT>
using Log2StridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, Log2Functor<argTy, resTy>>;

template <typename T> struct Log2OutputType
{
    using value_type = typename std::disjunction<
        td_ns::TypeMapResultEntry<T, sycl::half, sycl::half>,
        td_ns::TypeMapResultEntry<T, float, float>,
        td_ns::TypeMapResultEntry<T, double, double>,
        td_ns::TypeMapResultEntry<T, std::complex<float>, std::complex<float>>,
        td_ns::
            TypeMapResultEntry<T, std::complex<double>, std::complex<double>>,
        td_ns::DefaultResultEntry<void>>::result_type;

    static constexpr bool is_defined = !std::is_same_v<value_type, void>;
};

namespace hyperparam_detail
{

namespace vsu_ns = dpctl::tensor::kernels::vec_size_utils;

using vsu_ns::ContigHyperparameterSetDefault;
using vsu_ns::UnaryContigHyperparameterSetEntry;

template <typename argTy> struct Log2ContigHyperparameterSet
{
    using value_type =
        typename std::disjunction<ContigHyperparameterSetDefault<4u, 2u>>;

    constexpr static auto vec_sz = value_type::vec_sz;
    constexpr static auto n_vecs = value_type::n_vecs;
};

} // end of namespace hyperparam_detail

template <typename T1, typename T2, std::uint8_t vec_sz, std::uint8_t n_vecs>
class log2_contig_kernel;

template <typename argTy>
sycl::event log2_contig_impl(sycl::queue &exec_q,
                             std::size_t nelems,
                             const char *arg_p,
                             char *res_p,
                             const std::vector<sycl::event> &depends = {})
{
    using Log2HS = hyperparam_detail::Log2ContigHyperparameterSet<argTy>;
    static constexpr std::uint8_t vec_sz = Log2HS::vec_sz;
    static constexpr std::uint8_t n_vecs = Log2HS::n_vecs;

    return elementwise_common::unary_contig_impl<
        argTy, Log2OutputType, Log2ContigFunctor, log2_contig_kernel, vec_sz,
        n_vecs>(exec_q, nelems, arg_p, res_p, depends);
}

template <typename fnT, typename T> struct Log2ContigFactory
{
    fnT get()
    {
        if constexpr (!Log2OutputType<T>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = log2_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T> struct Log2TypeMapFactory
{
    /*! @brief get typeid for output type of sycl::log2(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename Log2OutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3> class log2_strided_kernel;

template <typename argTy>
sycl::event
log2_strided_impl(sycl::queue &exec_q,
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
        argTy, Log2OutputType, Log2StridedFunctor, log2_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T> struct Log2StridedFactory
{
    fnT get()
    {
        if constexpr (!Log2OutputType<T>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = log2_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace log2
} // namespace kernels
} // namespace tensor
} // namespace dpctl
