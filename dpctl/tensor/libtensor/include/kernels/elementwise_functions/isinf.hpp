//=== isinf.hpp -   Unary function ISINF                 ------  *-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of ISINF(x)
/// function that tests whether a tensor element is an infinity.
//===---------------------------------------------------------------------===//

#pragma once
#include <cmath>
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
namespace isinf
{

namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;
using dpctl::tensor::type_utils::vec_cast;

template <typename argT, typename resT> struct IsInfFunctor
{
    static_assert(std::is_same_v<resT, bool>);

    using is_constant = typename std::disjunction<std::is_same<argT, bool>,
                                                  std::is_integral<argT>>;
    static constexpr resT constant_value = false;
    using supports_vec =
        typename std::disjunction<std::is_same<argT, sycl::half>,
                                  std::is_floating_point<argT>>;
    using supports_sg_loadstore = typename std::negation<
        std::disjunction<is_complex<resT>, is_complex<argT>>>;

    resT operator()(const argT &in) const
    {
        if constexpr (is_complex<argT>::value) {
            const bool real_isinf = std::isinf(std::real(in));
            const bool imag_isinf = std::isinf(std::imag(in));
            return (real_isinf || imag_isinf);
        }
        else if constexpr (std::is_same<argT, bool>::value ||
                           std::is_integral<argT>::value)
        {
            return constant_value;
        }
        else if constexpr (std::is_same_v<argT, sycl::half>) {
            return sycl::isinf(in);
        }
        else {
            return std::isinf(in);
        }
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz> operator()(const sycl::vec<argT, vec_sz> &in) const
    {
        auto const &res_vec = sycl::isinf(in);

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
using IsInfContigFunctor =
    elementwise_common::UnaryContigFunctor<argT,
                                           resT,
                                           IsInfFunctor<argT, resT>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename argTy, typename resTy, typename IndexerT>
using IsInfStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, IsInfFunctor<argTy, resTy>>;

template <typename argTy> struct IsInfOutputType
{
    using value_type = bool;
};

template <typename T1, typename T2, unsigned int vec_sz, unsigned int n_vecs>
class isinf_contig_kernel;

template <typename argTy>
sycl::event isinf_contig_impl(sycl::queue &exec_q,
                              size_t nelems,
                              const char *arg_p,
                              char *res_p,
                              const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::unary_contig_impl<
        argTy, IsInfOutputType, IsInfContigFunctor, isinf_contig_kernel>(
        exec_q, nelems, arg_p, res_p, depends);
}

template <typename fnT, typename T> struct IsInfContigFactory
{
    fnT get()
    {
        fnT fn = isinf_contig_impl<T>;
        return fn;
    }
};

template <typename fnT, typename T> struct IsInfTypeMapFactory
{
    /*! @brief get typeid for output type of sycl::isinf(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename IsInfOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3> class isinf_strided_kernel;

template <typename argTy>
sycl::event
isinf_strided_impl(sycl::queue &exec_q,
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
        argTy, IsInfOutputType, IsInfStridedFunctor, isinf_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T> struct IsInfStridedFactory
{
    fnT get()
    {
        fnT fn = isinf_strided_impl<T>;
        return fn;
    }
};

} // namespace isinf
} // namespace kernels
} // namespace tensor
} // namespace dpctl
