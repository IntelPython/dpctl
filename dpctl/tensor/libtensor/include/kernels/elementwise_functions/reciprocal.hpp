//=== reciprocal.hpp -   Unary function RECIPROCAL                     ------
//*-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of RECIPROCAL(x)
/// function.
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

#include "utils/offset_utils.hpp"
#include "utils/type_dispatch_building.hpp"
#include "utils/type_utils.hpp"

#include "kernels/dpctl_tensor_types.hpp"
#include "kernels/elementwise_functions/common.hpp"

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace reciprocal
{

using dpctl::tensor::ssize_t;
namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;

template <typename argT, typename resT> struct ReciprocalFunctor
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

            return realT(1) / exprm_ns::complex<realT>(in);
        }
        else {
            return argT(1) / in;
        }
    }
};

template <typename argTy,
          typename resTy = argTy,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
using ReciprocalContigFunctor =
    elementwise_common::UnaryContigFunctor<argTy,
                                           resTy,
                                           ReciprocalFunctor<argTy, resTy>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename argTy, typename resTy, typename IndexerT>
using ReciprocalStridedFunctor =
    elementwise_common::UnaryStridedFunctor<argTy,
                                            resTy,
                                            IndexerT,
                                            ReciprocalFunctor<argTy, resTy>>;

template <typename T> struct ReciprocalOutputType
{
    using value_type = typename std::disjunction<
        td_ns::TypeMapResultEntry<T, sycl::half>,
        td_ns::TypeMapResultEntry<T, float>,
        td_ns::TypeMapResultEntry<T, double>,
        td_ns::TypeMapResultEntry<T, std::complex<float>>,
        td_ns::TypeMapResultEntry<T, std::complex<double>>,
        td_ns::DefaultResultEntry<void>>::result_type;

    static constexpr bool is_defined = !std::is_same_v<value_type, void>;
};

namespace
{

namespace vsu_ns = dpctl::tensor::kernels::vec_size_utils;

using vsu_ns::ContigHyperparameterSetDefault;
using vsu_ns::UnaryContigHyperparameterSetEntry;

template <typename argTy> struct ReciprocalContigHyperparameterSet
{
    using value_type =
        typename std::disjunction<ContigHyperparameterSetDefault<4u, 2u>>;

    constexpr static auto vec_sz = value_type::vec_sz;
    constexpr static auto n_vecs = value_type::n_vecs;
};

} // end of anonymous namespace

template <typename T1, typename T2, std::uint8_t vec_sz, std::uint8_t n_vecs>
class reciprocal_contig_kernel;

template <typename argTy>
sycl::event reciprocal_contig_impl(sycl::queue &exec_q,
                                   std::size_t nelems,
                                   const char *arg_p,
                                   char *res_p,
                                   const std::vector<sycl::event> &depends = {})
{
    constexpr std::uint8_t vec_sz =
        ReciprocalContigHyperparameterSet<argTy>::vec_sz;
    constexpr std::uint8_t n_vecs =
        ReciprocalContigHyperparameterSet<argTy>::n_vecs;

    return elementwise_common::unary_contig_impl<
        argTy, ReciprocalOutputType, ReciprocalContigFunctor,
        reciprocal_contig_kernel, vec_sz, n_vecs>(exec_q, nelems, arg_p, res_p,
                                                  depends);
}

template <typename fnT, typename T> struct ReciprocalContigFactory
{
    fnT get()
    {
        if constexpr (!ReciprocalOutputType<T>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = reciprocal_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T> struct ReciprocalTypeMapFactory
{
    /*! @brief get typeid for output type of 1 / x */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename ReciprocalOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3>
class reciprocal_strided_kernel;

template <typename argTy>
sycl::event
reciprocal_strided_impl(sycl::queue &exec_q,
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
    return elementwise_common::unary_strided_impl<argTy, ReciprocalOutputType,
                                                  ReciprocalStridedFunctor,
                                                  reciprocal_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T> struct ReciprocalStridedFactory
{
    fnT get()
    {
        if constexpr (!ReciprocalOutputType<T>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = reciprocal_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace reciprocal
} // namespace kernels
} // namespace tensor
} // namespace dpctl
