//=== NEXTAFTER.hpp -   Binary function NEXTAFTER  ------ *-C++-*--/===//
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
/// This file defines kernels for elementwise evaluation of NEXTAFTER(x1, x2)
/// function.
//===---------------------------------------------------------------------===//

#pragma once
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <type_traits>

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
namespace nextafter
{

using dpctl::tensor::ssize_t;
namespace td_ns = dpctl::tensor::type_dispatch;
namespace tu_ns = dpctl::tensor::type_utils;

template <typename argT1, typename argT2, typename resT> struct NextafterFunctor
{

    using supports_sg_loadstore = std::true_type;
    using supports_vec = std::true_type;

    resT operator()(const argT1 &in1, const argT2 &in2) const
    {
        return sycl::nextafter(in1, in2);
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz>
    operator()(const sycl::vec<argT1, vec_sz> &in1,
               const sycl::vec<argT2, vec_sz> &in2) const
    {
        auto res = sycl::nextafter(in1, in2);
        if constexpr (std::is_same_v<resT,
                                     typename decltype(res)::element_type>)
        {
            return res;
        }
        else {
            using dpctl::tensor::type_utils::vec_cast;

            return vec_cast<resT, typename decltype(res)::element_type, vec_sz>(
                res);
        }
    }
};

template <typename argT1,
          typename argT2,
          typename resT,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
using NextafterContigFunctor = elementwise_common::BinaryContigFunctor<
    argT1,
    argT2,
    resT,
    NextafterFunctor<argT1, argT2, resT>,
    vec_sz,
    n_vecs,
    enable_sg_loadstore>;

template <typename argT1, typename argT2, typename resT, typename IndexerT>
using NextafterStridedFunctor = elementwise_common::BinaryStridedFunctor<
    argT1,
    argT2,
    resT,
    IndexerT,
    NextafterFunctor<argT1, argT2, resT>>;

template <typename T1, typename T2> struct NextafterOutputType
{
    using value_type = typename std::disjunction<
        td_ns::BinaryTypeMapResultEntry<T1,
                                        sycl::half,
                                        T2,
                                        sycl::half,
                                        sycl::half>,
        td_ns::BinaryTypeMapResultEntry<T1, float, T2, float, float>,
        td_ns::BinaryTypeMapResultEntry<T1, double, T2, double, double>,
        td_ns::DefaultResultEntry<void>>::result_type;

    static constexpr bool is_defined = !std::is_same_v<value_type, void>;
};

namespace hyperparam_detail
{

namespace vsu_ns = dpctl::tensor::kernels::vec_size_utils;

using vsu_ns::BinaryContigHyperparameterSetEntry;
using vsu_ns::ContigHyperparameterSetDefault;

template <typename argTy1, typename argTy2>
struct NextafterContigHyperparameterSet
{
    using value_type =
        typename std::disjunction<ContigHyperparameterSetDefault<4u, 2u>>;

    constexpr static auto vec_sz = value_type::vec_sz;
    constexpr static auto n_vecs = value_type::n_vecs;
};

} // end of namespace hyperparam_detail

template <typename argT1,
          typename argT2,
          typename resT,
          std::uint8_t vec_sz,
          std::uint8_t n_vecs>
class nextafter_contig_kernel;

template <typename argTy1, typename argTy2>
sycl::event nextafter_contig_impl(sycl::queue &exec_q,
                                  std::size_t nelems,
                                  const char *arg1_p,
                                  ssize_t arg1_offset,
                                  const char *arg2_p,
                                  ssize_t arg2_offset,
                                  char *res_p,
                                  ssize_t res_offset,
                                  const std::vector<sycl::event> &depends = {})
{
    using NextafterHS =
        hyperparam_detail::NextafterContigHyperparameterSet<argTy1, argTy2>;
    constexpr std::uint8_t vec_sz = NextafterHS::vec_sz;
    constexpr std::uint8_t n_vecs = NextafterHS::n_vecs;

    return elementwise_common::binary_contig_impl<
        argTy1, argTy2, NextafterOutputType, NextafterContigFunctor,
        nextafter_contig_kernel, vec_sz, n_vecs>(
        exec_q, nelems, arg1_p, arg1_offset, arg2_p, arg2_offset, res_p,
        res_offset, depends);
}

template <typename fnT, typename T1, typename T2> struct NextafterContigFactory
{
    fnT get()
    {
        if constexpr (!NextafterOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = nextafter_contig_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2> struct NextafterTypeMapFactory
{
    /*! @brief get typeid for output type of std::nextafter(T1 x, T2 y) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename NextafterOutputType<T1, T2>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename resT, typename IndexerT>
class nextafter_strided_kernel;

template <typename argTy1, typename argTy2>
sycl::event
nextafter_strided_impl(sycl::queue &exec_q,
                       std::size_t nelems,
                       int nd,
                       const ssize_t *shape_and_strides,
                       const char *arg1_p,
                       ssize_t arg1_offset,
                       const char *arg2_p,
                       ssize_t arg2_offset,
                       char *res_p,
                       ssize_t res_offset,
                       const std::vector<sycl::event> &depends,
                       const std::vector<sycl::event> &additional_depends)
{
    return elementwise_common::binary_strided_impl<
        argTy1, argTy2, NextafterOutputType, NextafterStridedFunctor,
        nextafter_strided_kernel>(exec_q, nelems, nd, shape_and_strides, arg1_p,
                                  arg1_offset, arg2_p, arg2_offset, res_p,
                                  res_offset, depends, additional_depends);
}

template <typename fnT, typename T1, typename T2> struct NextafterStridedFactory
{
    fnT get()
    {
        if constexpr (!NextafterOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = nextafter_strided_impl<T1, T2>;
            return fn;
        }
    }
};

} // namespace nextafter
} // namespace kernels
} // namespace tensor
} // namespace dpctl
