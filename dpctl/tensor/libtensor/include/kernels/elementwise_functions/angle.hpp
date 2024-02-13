//=== angle.hpp -   Unary function ANGLE              ------*-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of ANGLE(x) function.
//===---------------------------------------------------------------------===//

#pragma once
#include <cmath>
#include <complex>
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
namespace angle
{

namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;

template <typename argT, typename resT> struct AngleFunctor
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
#ifdef USE_SYCL_FOR_COMPLEX_TYPES
        using rT = typename argT::value_type;

        return exprm_ns::arg(exprm_ns::complex<rT>(in)); // std::arg(in);
#else
        return std::arg(in);
#endif
    }
};

template <typename argTy,
          typename resTy = argTy,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using AngleContigFunctor =
    elementwise_common::UnaryContigFunctor<argTy,
                                           resTy,
                                           AngleFunctor<argTy, resTy>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename argTy, typename resTy, typename IndexerT>
using AngleStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, AngleFunctor<argTy, resTy>>;

template <typename T> struct AngleOutputType
{
    using value_type = typename std::disjunction< // disjunction is C++17
                                                  // feature, supported by DPC++
        td_ns::TypeMapResultEntry<T, std::complex<float>, float>,
        td_ns::TypeMapResultEntry<T, std::complex<double>, double>,
        td_ns::DefaultResultEntry<void>>::result_type;
};

template <typename T1, typename T2, unsigned int vec_sz, unsigned int n_vecs>
class angle_contig_kernel;

template <typename argTy>
sycl::event angle_contig_impl(sycl::queue &exec_q,
                              size_t nelems,
                              const char *arg_p,
                              char *res_p,
                              const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::unary_contig_impl<
        argTy, AngleOutputType, AngleContigFunctor, angle_contig_kernel>(
        exec_q, nelems, arg_p, res_p, depends);
}

template <typename fnT, typename T> struct AngleContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename AngleOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = angle_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T> struct AngleTypeMapFactory
{
    /*! @brief get typeid for output type of std::arg(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename AngleOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3> class angle_strided_kernel;

template <typename argTy>
sycl::event
angle_strided_impl(sycl::queue &exec_q,
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
        argTy, AngleOutputType, AngleStridedFunctor, angle_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T> struct AngleStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename AngleOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = angle_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace angle
} // namespace kernels
} // namespace tensor
} // namespace dpctl
