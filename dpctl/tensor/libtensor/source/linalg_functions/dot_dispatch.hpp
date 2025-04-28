//===-- ------------ Implementation of _tensor_impl module  ----*-C++-*-/===//
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
//===--------------------------------------------------------------------===//
///
/// \file
/// This file defines functions of dpctl.tensor._tensor_impl extensions
//===--------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

#include "kernels/linalg_functions/dot_product.hpp"
#include "kernels/linalg_functions/gemm.hpp"
#include "utils/type_dispatch_building.hpp"

#define SYCL_EXT_ONEAPI_COMPLEX
#include <sycl/ext/oneapi/experimental/complex/complex.hpp>

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

namespace td_ns = dpctl::tensor::type_dispatch;
namespace exprm_ns = sycl::ext::oneapi::experimental;

template <typename T1, typename T2> struct DotAtomicOutputType
{
    using value_type = typename std::disjunction<
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint32_t,
                                        T2,
                                        std::uint32_t,
                                        std::uint32_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint32_t,
                                        T2,
                                        std::uint32_t,
                                        std::uint64_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::int32_t,
                                        T2,
                                        std::int32_t,
                                        std::int32_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::int32_t,
                                        T2,
                                        std::int32_t,
                                        std::int64_t>,
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
        td_ns::BinaryTypeMapResultEntry<T1, float, T2, float, float>,
        td_ns::BinaryTypeMapResultEntry<T1, float, T2, float, double>,
        td_ns::BinaryTypeMapResultEntry<T1, double, T2, double, double>,
        td_ns::DefaultResultEntry<void>>::result_type;

    static constexpr bool is_defined = !std::is_same_v<value_type, void>;
};

// add separate type support lists for atomic vs. temps
// gemm, gevm, and dot product share output type struct
template <typename T1, typename T2> struct DotNoAtomicOutputType
{
    using value_type = typename std::disjunction<
        td_ns::BinaryTypeMapResultEntry<T1, bool, T2, bool, bool>,
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
                                        std::uint32_t,
                                        T2,
                                        std::uint32_t,
                                        std::uint64_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::int32_t,
                                        T2,
                                        std::int32_t,
                                        std::int32_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::int32_t,
                                        T2,
                                        std::int32_t,
                                        std::int64_t>,
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
        td_ns::BinaryTypeMapResultEntry<T1,
                                        exprm_ns::complex<float>,
                                        T2,
                                        exprm_ns::complex<float>,
                                        exprm_ns::complex<float>>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        exprm_ns::complex<float>,
                                        T2,
                                        exprm_ns::complex<float>,
                                        exprm_ns::complex<double>>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        exprm_ns::complex<double>,
                                        T2,
                                        exprm_ns::complex<double>,
                                        exprm_ns::complex<double>>,
        td_ns::DefaultResultEntry<void>>::result_type;

    static constexpr bool is_defined = !std::is_same_v<value_type, void>;
};

template <typename fnT, typename T1, typename T2> struct DotTypeMapFactory
{
    /*! @brief get typeid for output type of kernels called by py_dot */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT1 = typename DotNoAtomicOutputType<T1, T2>::value_type;
        using rT2 = typename DotAtomicOutputType<T1, T2>::value_type;
        static_assert(std::is_same_v<rT1, rT2> || std::is_same_v<rT2, void>);
        return td_ns::GetTypeid<rT1>{}.get();
    }
};

template <typename fnT, typename T1, typename T2> struct GemmBatchAtomicFactory
{
    fnT get()
    {
        if constexpr (!DotAtomicOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            using dpctl::tensor::kernels::gemm_batch_impl;
            using T3 = typename DotAtomicOutputType<T1, T2>::value_type;
            fnT fn = gemm_batch_impl<T1, T2, T3>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2>
struct GemmBatchContigAtomicFactory
{
    fnT get()
    {
        if constexpr (!DotAtomicOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            using dpctl::tensor::kernels::gemm_batch_contig_impl;
            using T3 = typename DotAtomicOutputType<T1, T2>::value_type;
            fnT fn = gemm_batch_contig_impl<T1, T2, T3>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2> struct GemmAtomicFactory
{
    fnT get()
    {
        if constexpr (!DotAtomicOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            using dpctl::tensor::kernels::gemm_impl;
            using T3 = typename DotAtomicOutputType<T1, T2>::value_type;
            fnT fn = gemm_impl<T1, T2, T3>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2> struct GemmContigAtomicFactory
{
    fnT get()
    {
        if constexpr (!DotAtomicOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            using dpctl::tensor::kernels::gemm_contig_impl;
            using T3 = typename DotAtomicOutputType<T1, T2>::value_type;
            fnT fn = gemm_contig_impl<T1, T2, T3>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2> struct GemmTempsFactory
{
    fnT get()
    {
        if constexpr (!DotNoAtomicOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            using dpctl::tensor::kernels::gemm_tree_impl;
            using T3 = typename DotNoAtomicOutputType<T1, T2>::value_type;
            fnT fn = gemm_tree_impl<T1, T2, T3>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2> struct GemmContigTempsFactory
{
    fnT get()
    {
        if constexpr (!DotNoAtomicOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            using dpctl::tensor::kernels::gemm_contig_tree_impl;
            using T3 = typename DotNoAtomicOutputType<T1, T2>::value_type;
            fnT fn = gemm_contig_tree_impl<T1, T2, T3>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2> struct GemmBatchTempsFactory
{
    fnT get()
    {
        if constexpr (!DotNoAtomicOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            using dpctl::tensor::kernels::gemm_batch_tree_impl;
            using T3 = typename DotNoAtomicOutputType<T1, T2>::value_type;
            fnT fn = gemm_batch_tree_impl<T1, T2, T3>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2>
struct GemmBatchContigTempsFactory
{
    fnT get()
    {
        if constexpr (!DotNoAtomicOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            using dpctl::tensor::kernels::gemm_batch_contig_tree_impl;
            using T3 = typename DotNoAtomicOutputType<T1, T2>::value_type;
            fnT fn = gemm_batch_contig_tree_impl<T1, T2, T3>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2> struct DotProductAtomicFactory
{
    fnT get()
    {
        if constexpr (!DotAtomicOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            using dpctl::tensor::kernels::dot_product_impl;
            using T3 = typename DotAtomicOutputType<T1, T2>::value_type;
            fnT fn = dot_product_impl<T1, T2, T3>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2>
struct DotProductNoAtomicFactory
{
    fnT get()
    {
        if constexpr (!DotNoAtomicOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            using dpctl::tensor::kernels::dot_product_tree_impl;
            using T3 = typename DotNoAtomicOutputType<T1, T2>::value_type;
            fnT fn = dot_product_tree_impl<T1, T2, T3>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2>
struct DotProductContigAtomicFactory
{
    fnT get()
    {
        if constexpr (!DotAtomicOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            using dpctl::tensor::kernels::dot_product_contig_impl;
            using T3 = typename DotAtomicOutputType<T1, T2>::value_type;
            fnT fn = dot_product_contig_impl<T1, T2, T3>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2>
struct DotProductContigNoAtomicFactory
{
    fnT get()
    {
        if constexpr (!DotNoAtomicOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            using dpctl::tensor::kernels::dot_product_contig_tree_impl;
            using T3 = typename DotNoAtomicOutputType<T1, T2>::value_type;
            fnT fn = dot_product_contig_tree_impl<T1, T2, T3>;
            return fn;
        }
    }
};

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
