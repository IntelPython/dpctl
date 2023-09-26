//=== floor_divide.hpp -  Binary function FLOOR_DIVIDE  ------  *-C++-*--/===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2023 Intel Corporation
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
/// This file defines kernels for elementwise evaluation of FLOOR_DIVIDE(x1, x2)
/// function.
//===---------------------------------------------------------------------===//

#pragma once
#include <CL/sycl.hpp>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "utils/offset_utils.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

#include "kernels/elementwise_functions/common.hpp"
#include <pybind11/pybind11.h>

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace floor_divide
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;
namespace tu_ns = dpctl::tensor::type_utils;

template <typename argT1, typename argT2, typename resT>
struct FloorDivideFunctor
{
    using supports_sg_loadstore = std::true_type;
    using supports_vec = std::true_type;

    resT operator()(const argT1 &in1, const argT2 &in2) const
    {
        if constexpr (std::is_same_v<argT1, bool> &&
                      std::is_same_v<argT2, bool>) {
            return (in2) ? static_cast<resT>(in1) : resT(0);
        }
        else if constexpr (std::is_integral_v<argT1> ||
                           std::is_integral_v<argT2>) {
            if (in2 == argT2(0)) {
                return resT(0);
            }
            if constexpr (std::is_signed_v<argT1> || std::is_signed_v<argT2>) {
                auto div = in1 / in2;
                auto mod = in1 % in2;
                auto corr = (mod != 0 && l_xor(mod < 0, in2 < 0));
                return (div - corr);
            }
            else {
                return (in1 / in2);
            }
        }
        else {
            auto div = in1 / in2;
            return (div == resT(0)) ? div : resT(std::floor(div));
        }
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz>
    operator()(const sycl::vec<argT1, vec_sz> &in1,
               const sycl::vec<argT2, vec_sz> &in2) const
    {
        if constexpr (std::is_same_v<argT1, bool> &&
                      std::is_same_v<argT2, bool>) {
            sycl::vec<resT, vec_sz> res;
#pragma unroll
            for (int i = 0; i < vec_sz; ++i) {
                res[i] = (in2[i]) ? static_cast<resT>(in1[i]) : resT(0);
            }
            return res;
        }
        else if constexpr (std::is_integral_v<resT>) {
            sycl::vec<resT, vec_sz> res;
#pragma unroll
            for (int i = 0; i < vec_sz; ++i) {
                if (in2[i] == argT2(0)) {
                    res[i] = resT(0);
                }
                else {
                    res[i] = in1[i] / in2[i];
                    if constexpr (std::is_signed_v<resT>) {
                        auto mod = in1[i] % in2[i];
                        auto corr = (mod != 0 && l_xor(mod < 0, in2[i] < 0));
                        res[i] -= corr;
                    }
                }
            }
            return res;
        }
        else {
            auto tmp = in1 / in2;
            using tmpT = typename decltype(tmp)::element_type;
#pragma unroll
            for (int i = 0; i < vec_sz; ++i) {
                if (in2[i] != argT2(0)) {
                    tmp[i] = std::floor(tmp[i]);
                }
            }
            if constexpr (std::is_same_v<resT, tmpT>) {
                return tmp;
            }
            else {
                using dpctl::tensor::type_utils::vec_cast;
                return vec_cast<resT, tmpT, vec_sz>(tmp);
            }
        }
    }

private:
    bool l_xor(bool b1, bool b2) const
    {
        return (b1 != b2);
    }
};

template <typename argT1,
          typename argT2,
          typename resT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2>
using FloorDivideContigFunctor = elementwise_common::BinaryContigFunctor<
    argT1,
    argT2,
    resT,
    FloorDivideFunctor<argT1, argT2, resT>,
    vec_sz,
    n_vecs>;

template <typename argT1, typename argT2, typename resT, typename IndexerT>
using FloorDivideStridedFunctor = elementwise_common::BinaryStridedFunctor<
    argT1,
    argT2,
    resT,
    IndexerT,
    FloorDivideFunctor<argT1, argT2, resT>>;

template <typename T1, typename T2> struct FloorDivideOutputType
{
    using value_type = typename std::disjunction< // disjunction is C++17
                                                  // feature, supported by DPC++
        td_ns::BinaryTypeMapResultEntry<T1, bool, T2, bool, std::int8_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint8_t,
                                        T2,
                                        std::uint8_t,
                                        std::uint8_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::int8_t,
                                        T2,
                                        std::int8_t,
                                        std::int8_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint16_t,
                                        T2,
                                        std::uint16_t,
                                        std::uint16_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::int16_t,
                                        T2,
                                        std::int16_t,
                                        std::int16_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint32_t,
                                        T2,
                                        std::uint32_t,
                                        std::uint32_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::int32_t,
                                        T2,
                                        std::int32_t,
                                        std::int32_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint64_t,
                                        T2,
                                        std::uint64_t,
                                        std::uint64_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::int64_t,
                                        T2,
                                        std::int64_t,
                                        std::int64_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        sycl::half,
                                        T2,
                                        sycl::half,
                                        sycl::half>,
        td_ns::BinaryTypeMapResultEntry<T1, float, T2, float, float>,
        td_ns::BinaryTypeMapResultEntry<T1, double, T2, double, double>,
        td_ns::DefaultResultEntry<void>>::result_type;
};

template <typename argT1,
          typename argT2,
          typename resT,
          unsigned int vec_sz,
          unsigned int n_vecs>
class floor_divide_contig_kernel;

template <typename argTy1, typename argTy2>
sycl::event
floor_divide_contig_impl(sycl::queue &exec_q,
                         size_t nelems,
                         const char *arg1_p,
                         py::ssize_t arg1_offset,
                         const char *arg2_p,
                         py::ssize_t arg2_offset,
                         char *res_p,
                         py::ssize_t res_offset,
                         const std::vector<sycl::event> &depends = {})
{
    return elementwise_common::binary_contig_impl<
        argTy1, argTy2, FloorDivideOutputType, FloorDivideContigFunctor,
        floor_divide_contig_kernel>(exec_q, nelems, arg1_p, arg1_offset, arg2_p,
                                    arg2_offset, res_p, res_offset, depends);
}

template <typename fnT, typename T1, typename T2>
struct FloorDivideContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename FloorDivideOutputType<T1, T2>::value_type,
                          void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = floor_divide_contig_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2>
struct FloorDivideTypeMapFactory
{
    /*! @brief get typeid for output type of floor_divide(T1 x, T2 y) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename FloorDivideOutputType<T1, T2>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename resT, typename IndexerT>
class floor_divide_strided_kernel;

template <typename argTy1, typename argTy2>
sycl::event
floor_divide_strided_impl(sycl::queue &exec_q,
                          size_t nelems,
                          int nd,
                          const py::ssize_t *shape_and_strides,
                          const char *arg1_p,
                          py::ssize_t arg1_offset,
                          const char *arg2_p,
                          py::ssize_t arg2_offset,
                          char *res_p,
                          py::ssize_t res_offset,
                          const std::vector<sycl::event> &depends,
                          const std::vector<sycl::event> &additional_depends)
{
    return elementwise_common::binary_strided_impl<
        argTy1, argTy2, FloorDivideOutputType, FloorDivideStridedFunctor,
        floor_divide_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg1_p, arg1_offset, arg2_p,
        arg2_offset, res_p, res_offset, depends, additional_depends);
}

template <typename fnT, typename T1, typename T2>
struct FloorDivideStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename FloorDivideOutputType<T1, T2>::value_type,
                          void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = floor_divide_strided_impl<T1, T2>;
            return fn;
        }
    }
};

} // namespace floor_divide
} // namespace kernels
} // namespace tensor
} // namespace dpctl
