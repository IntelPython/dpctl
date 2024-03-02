//===--type_dispatch_building.cpp - Type-dispatch table building utils -*-C++-*-
//===//
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
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines class to implement dispatch tables for pair of types
//===----------------------------------------------------------------------===//

#pragma once

#include <complex>
#include <sycl/sycl.hpp>

namespace dpctl
{
namespace tensor
{

namespace type_dispatch
{

enum class typenum_t : int
{
    BOOL = 0,
    INT8, // 1
    UINT8,
    INT16,
    UINT16,
    INT32, // 5
    UINT32,
    INT64,
    UINT64,
    HALF,
    FLOAT, // 10
    DOUBLE,
    CFLOAT,
    CDOUBLE, // 13
};
constexpr int num_types = 14; // number of elements in typenum_t

template <typename funcPtrT,
          template <typename fnT, typename D, typename S>
          typename factory,
          int _num_types>
class DispatchTableBuilder
{
private:
    template <typename dstTy>
    const std::vector<funcPtrT> row_per_dst_type() const
    {
        std::vector<funcPtrT> per_dstTy = {
            factory<funcPtrT, dstTy, bool>{}.get(),
            factory<funcPtrT, dstTy, int8_t>{}.get(),
            factory<funcPtrT, dstTy, uint8_t>{}.get(),
            factory<funcPtrT, dstTy, int16_t>{}.get(),
            factory<funcPtrT, dstTy, uint16_t>{}.get(),
            factory<funcPtrT, dstTy, int32_t>{}.get(),
            factory<funcPtrT, dstTy, uint32_t>{}.get(),
            factory<funcPtrT, dstTy, int64_t>{}.get(),
            factory<funcPtrT, dstTy, uint64_t>{}.get(),
            factory<funcPtrT, dstTy, sycl::half>{}.get(),
            factory<funcPtrT, dstTy, float>{}.get(),
            factory<funcPtrT, dstTy, double>{}.get(),
            factory<funcPtrT, dstTy, std::complex<float>>{}.get(),
            factory<funcPtrT, dstTy, std::complex<double>>{}.get()};
        assert(per_dstTy.size() == _num_types);
        return per_dstTy;
    }

public:
    DispatchTableBuilder() = default;
    ~DispatchTableBuilder() = default;

    void populate_dispatch_table(funcPtrT table[][_num_types]) const
    {
        const auto map_by_dst_type = {row_per_dst_type<bool>(),
                                      row_per_dst_type<int8_t>(),
                                      row_per_dst_type<uint8_t>(),
                                      row_per_dst_type<int16_t>(),
                                      row_per_dst_type<uint16_t>(),
                                      row_per_dst_type<int32_t>(),
                                      row_per_dst_type<uint32_t>(),
                                      row_per_dst_type<int64_t>(),
                                      row_per_dst_type<uint64_t>(),
                                      row_per_dst_type<sycl::half>(),
                                      row_per_dst_type<float>(),
                                      row_per_dst_type<double>(),
                                      row_per_dst_type<std::complex<float>>(),
                                      row_per_dst_type<std::complex<double>>()};
        assert(map_by_dst_type.size() == _num_types);
        int dst_id = 0;
        for (auto &row : map_by_dst_type) {
            int src_id = 0;
            for (auto &fn_ptr : row) {
                table[dst_id][src_id] = fn_ptr;
                ++src_id;
            }
            ++dst_id;
        }
    }
};

template <typename funcPtrT,
          template <typename fnT, typename T>
          typename factory,
          int _num_types>
class DispatchVectorBuilder
{
private:
    template <typename Ty> const funcPtrT func_per_type() const
    {
        funcPtrT f = factory<funcPtrT, Ty>{}.get();
        return f;
    }

public:
    DispatchVectorBuilder() = default;
    ~DispatchVectorBuilder() = default;

    void populate_dispatch_vector(funcPtrT vector[]) const
    {
        const auto fn_map_by_type = {func_per_type<bool>(),
                                     func_per_type<int8_t>(),
                                     func_per_type<uint8_t>(),
                                     func_per_type<int16_t>(),
                                     func_per_type<uint16_t>(),
                                     func_per_type<int32_t>(),
                                     func_per_type<uint32_t>(),
                                     func_per_type<int64_t>(),
                                     func_per_type<uint64_t>(),
                                     func_per_type<sycl::half>(),
                                     func_per_type<float>(),
                                     func_per_type<double>(),
                                     func_per_type<std::complex<float>>(),
                                     func_per_type<std::complex<double>>()};
        assert(fn_map_by_type.size() == _num_types);
        int ty_id = 0;
        for (auto &fn : fn_map_by_type) {
            vector[ty_id] = fn;
            ++ty_id;
        }
    }
};

/*! @brief struct to define result_type typename for Ty == ArgTy */
template <typename Ty, typename ArgTy, typename ResTy = ArgTy>
struct TypeMapResultEntry : std::bool_constant<std::is_same_v<Ty, ArgTy>>
{
    using result_type = ResTy;
};

/*! @brief struct to define result_type typename for Ty1 == ArgTy1 && Ty2 ==
 * ArgTy2 */
template <typename Ty1,
          typename ArgTy1,
          typename Ty2,
          typename ArgTy2,
          typename ResTy>
struct BinaryTypeMapResultEntry
    : std::bool_constant<std::conjunction_v<std::is_same<Ty1, ArgTy1>,
                                            std::is_same<Ty2, ArgTy2>>>
{
    using result_type = ResTy;
};

/*! @brief fall-through struct with specified result_type, usually void */
template <typename Ty = void> struct DefaultResultEntry : std::true_type
{
    using result_type = Ty;
};

/*! @brief Utility struct to convert C++ type into typeid integer */
template <typename T> struct GetTypeid
{
    int get()
    {
        if constexpr (std::is_same_v<T, bool>) {
            return static_cast<int>(typenum_t::BOOL);
        }
        else if constexpr (std::is_same_v<T, std::int8_t>) {
            return static_cast<int>(typenum_t::INT8);
        }
        else if constexpr (std::is_same_v<T, std::uint8_t>) {
            return static_cast<int>(typenum_t::UINT8);
        }
        else if constexpr (std::is_same_v<T, std::int16_t>) {
            return static_cast<int>(typenum_t::INT16);
        }
        else if constexpr (std::is_same_v<T, std::uint16_t>) {
            return static_cast<int>(typenum_t::UINT16);
        }
        else if constexpr (std::is_same_v<T, std::int32_t>) {
            return static_cast<int>(typenum_t::INT32);
        }
        else if constexpr (std::is_same_v<T, std::uint32_t>) {
            return static_cast<int>(typenum_t::UINT32);
        }
        else if constexpr (std::is_same_v<T, std::int64_t>) {
            return static_cast<int>(typenum_t::INT64);
        }
        else if constexpr (std::is_same_v<T, std::uint64_t>) {
            return static_cast<int>(typenum_t::UINT64);
        }
        else if constexpr (std::is_same_v<T, sycl::half>) {
            return static_cast<int>(typenum_t::HALF);
        }
        else if constexpr (std::is_same_v<T, float>) {
            return static_cast<int>(typenum_t::FLOAT);
        }
        else if constexpr (std::is_same_v<T, double>) {
            return static_cast<int>(typenum_t::DOUBLE);
        }
        else if constexpr (std::is_same_v<T, std::complex<float>>) {
            return static_cast<int>(typenum_t::CFLOAT);
        }
        else if constexpr (std::is_same_v<T, std::complex<double>>) {
            return static_cast<int>(typenum_t::CDOUBLE);
        }
        else if constexpr (std::is_same_v<T, void>) { // special token
            return -1;
        }

        assert(("Unsupported type T", false));
        return -2;
    }
};

/*! @brief Class to generate vector of null function pointers */
template <typename FunPtrT> struct NullPtrVector
{

    using value_type = FunPtrT;
    using const_reference = value_type const &;

    NullPtrVector() : val(nullptr) {}

    const_reference operator[](int) const
    {
        return val;
    }

private:
    value_type val;
};

/*! @brief Class to generate table of null function pointers */
template <typename FunPtrT> struct NullPtrTable
{
    using value_type = NullPtrVector<FunPtrT>;
    using const_reference = value_type const &;

    NullPtrTable() : val() {}

    const_reference operator[](int) const
    {
        return val;
    }

private:
    value_type val;
};

template <typename Ty1, typename ArgTy, typename Ty2, typename outTy>
struct TypePairDefinedEntry : std::bool_constant<std::is_same_v<Ty1, ArgTy> &&
                                                 std::is_same_v<Ty2, outTy>>
{
    static constexpr bool is_defined = true;
};

struct NotDefinedEntry : std::true_type
{
    static constexpr bool is_defined = false;
};

} // namespace type_dispatch

} // namespace tensor
} // namespace dpctl
