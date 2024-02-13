//=== isnan.hpp -   Unary function ISNAN                 ------  *-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of ISNAN(x)
/// function that tests whether a tensor element is a NaN.
//===---------------------------------------------------------------------===//

#pragma once
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <type_traits>

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
namespace isnan
{

namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;
using dpctl::tensor::type_utils::vec_cast;

template <typename argT, typename resT> struct IsNanFunctor
{
    static_assert(std::is_same_v<resT, bool>);

    /*
    std::is_same<argT, bool>::value ||
                           std::is_integral<argT>::value
    */
    using is_constant = typename std::disjunction<std::is_same<argT, bool>,
                                                  std::is_integral<argT>>;
    static constexpr resT constant_value = false;
    using supports_vec = typename std::true_type;
    using supports_sg_loadstore = typename std::negation<
        std::disjunction<is_complex<resT>, is_complex<argT>>>;

    resT operator()(const argT &in) const
    {
        if constexpr (is_complex<argT>::value) {
            const bool real_isnan = sycl::isnan(std::real(in));
            const bool imag_isnan = sycl::isnan(std::imag(in));
            return (real_isnan || imag_isnan);
        }
        else if constexpr (std::is_same<argT, bool>::value ||
                           std::is_integral<argT>::value)
        {
            return constant_value;
        }
        else {
            return sycl::isnan(in);
        }
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz> operator()(const sycl::vec<argT, vec_sz> &in) const
    {
        auto const &res_vec = sycl::isnan(in);

        using deducedT = typename std::remove_cv_t<
            std::remove_reference_t<decltype(res_vec)>>::element_type;

        return vec_cast<bool, deducedT, vec_sz>(res_vec);
    }
};

template <typename argT,
          typename resT = bool,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using IsNanContigFunctor =
    elementwise_common::UnaryContigFunctor<argT,
                                           resT,
                                           IsNanFunctor<argT, resT>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename argTy, typename resTy, typename IndexerT>
using IsNanStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, IsNanFunctor<argTy, resTy>>;

template <typename argTy> struct IsNanOutputType
{
    using value_type = bool;
};

template <typename T1, typename T2, unsigned int vec_sz, unsigned int n_vecs>
class isnan_contig_kernel;

template <typename argTy>
sycl::event isnan_contig_impl(sycl::queue &exec_q,
                              size_t nelems,
                              const char *arg_p,
                              char *res_p,
                              const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::unary_contig_impl<
        argTy, IsNanOutputType, IsNanContigFunctor, isnan_contig_kernel>(
        exec_q, nelems, arg_p, res_p, depends);
}

template <typename fnT, typename T> struct IsNanContigFactory
{
    fnT get()
    {
        fnT fn = isnan_contig_impl<T>;
        return fn;
    }
};

template <typename fnT, typename T> struct IsNanTypeMapFactory
{
    /*! @brief get typeid for output type of sycl::isnan(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename IsNanOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3> class isnan_strided_kernel;

template <typename argTy>
sycl::event
isnan_strided_impl(sycl::queue &exec_q,
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
        argTy, IsNanOutputType, IsNanStridedFunctor, isnan_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T> struct IsNanStridedFactory
{
    fnT get()
    {
        fnT fn = isnan_strided_impl<T>;
        return fn;
    }
};

} // namespace isnan
} // namespace kernels
} // namespace tensor
} // namespace dpctl
