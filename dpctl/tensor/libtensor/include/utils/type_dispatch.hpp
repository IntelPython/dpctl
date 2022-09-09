//===--type_dispatch.cpp - Type-dispatch table building utils ----*-C++-*- ===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2022 Intel Corporation
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

#include "dpctl4pybind11.hpp"
#include <CL/sycl.hpp>
#include <complex>

namespace dpctl
{
namespace tensor
{

namespace detail
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

// Lookup a type according to its size, and return a value corresponding to the
// NumPy typenum.
template <typename Concrete> constexpr int platform_typeid_lookup()
{
    return -1;
}

template <typename Concrete, typename T, typename... Ts, typename... Ints>
constexpr int platform_typeid_lookup(int I, Ints... Is)
{
    return sizeof(Concrete) == sizeof(T)
               ? I
               : platform_typeid_lookup<Concrete, Ts...>(Is...);
}

struct usm_ndarray_types
{
    static usm_ndarray_types &get()
    {
        static usm_ndarray_types singleton = populate_fields();
        return singleton;
    }

    int typenum_to_lookup_id(int typenum)
    {
        using typenum_t = dpctl::tensor::detail::typenum_t;

        if (typenum == UAR_DOUBLE_) {
            return static_cast<int>(typenum_t::DOUBLE);
        }
        else if (typenum == UAR_INT64_) {
            return static_cast<int>(typenum_t::INT64);
        }
        else if (typenum == UAR_INT32_) {
            return static_cast<int>(typenum_t::INT32);
        }
        else if (typenum == UAR_BOOL_) {
            return static_cast<int>(typenum_t::BOOL);
        }
        else if (typenum == UAR_CDOUBLE_) {
            return static_cast<int>(typenum_t::CDOUBLE);
        }
        else if (typenum == UAR_FLOAT_) {
            return static_cast<int>(typenum_t::FLOAT);
        }
        else if (typenum == UAR_INT16_) {
            return static_cast<int>(typenum_t::INT16);
        }
        else if (typenum == UAR_INT8_) {
            return static_cast<int>(typenum_t::INT8);
        }
        else if (typenum == UAR_UINT64_) {
            return static_cast<int>(typenum_t::UINT64);
        }
        else if (typenum == UAR_UINT32_) {
            return static_cast<int>(typenum_t::UINT32);
        }
        else if (typenum == UAR_UINT16_) {
            return static_cast<int>(typenum_t::UINT16);
        }
        else if (typenum == UAR_UINT8_) {
            return static_cast<int>(typenum_t::UINT8);
        }
        else if (typenum == UAR_CFLOAT_) {
            return static_cast<int>(typenum_t::CFLOAT);
        }
        else if (typenum == UAR_HALF_) {
            return static_cast<int>(typenum_t::HALF);
        }
        else if (typenum == UAR_INT || typenum == UAR_UINT) {
            switch (sizeof(int)) {
            case sizeof(std::int32_t):
                return ((typenum == UAR_INT)
                            ? static_cast<int>(typenum_t::INT32)
                            : static_cast<int>(typenum_t::UINT32));
            case sizeof(std::int64_t):
                return ((typenum == UAR_INT)
                            ? static_cast<int>(typenum_t::INT64)
                            : static_cast<int>(typenum_t::UINT64));
            default:
                throw_unrecognized_typenum_error(typenum);
            }
        }
        else {
            throw_unrecognized_typenum_error(typenum);
        }
        // return code signalling error, should never be reached
        assert(false);
        return -1;
    }

private:
    int UAR_BOOL_ = -1;
    // Platform-dependent normalization
    int UAR_INT8_ = -1;
    int UAR_UINT8_ = -1;
    int UAR_INT16_ = -1;
    int UAR_UINT16_ = -1;
    int UAR_INT32_ = -1;
    int UAR_UINT32_ = -1;
    int UAR_INT64_ = -1;
    int UAR_UINT64_ = -1;
    int UAR_HALF_ = -1;
    int UAR_FLOAT_ = -1;
    int UAR_DOUBLE_ = -1;
    int UAR_CFLOAT_ = -1;
    int UAR_CDOUBLE_ = -1;
    int UAR_TYPE_SENTINEL_ = -1;

    void init_constants()
    {
        UAR_BOOL_ = UAR_BOOL;
        UAR_INT8_ = UAR_BYTE;
        UAR_UINT8_ = UAR_UBYTE;
        UAR_INT16_ = UAR_SHORT;
        UAR_UINT16_ = UAR_USHORT;
        UAR_INT32_ = platform_typeid_lookup<std::int32_t, long, int, short>(
            UAR_LONG, UAR_INT, UAR_SHORT);
        UAR_UINT32_ = platform_typeid_lookup<std::uint32_t, unsigned long,
                                             unsigned int, unsigned short>(
            UAR_ULONG, UAR_UINT, UAR_USHORT);
        UAR_INT64_ = platform_typeid_lookup<std::int64_t, long, long long, int>(
            UAR_LONG, UAR_LONGLONG, UAR_INT);
        UAR_UINT64_ = platform_typeid_lookup<std::uint64_t, unsigned long,
                                             unsigned long long, unsigned int>(
            UAR_ULONG, UAR_ULONGLONG, UAR_UINT);
        UAR_HALF_ = UAR_HALF;
        UAR_FLOAT_ = UAR_FLOAT;
        UAR_DOUBLE_ = UAR_DOUBLE;
        UAR_CFLOAT_ = UAR_CFLOAT;
        UAR_CDOUBLE_ = UAR_CDOUBLE;
        UAR_TYPE_SENTINEL_ = UAR_TYPE_SENTINEL;
    }

    static usm_ndarray_types populate_fields()
    {
        import_dpctl();

        usm_ndarray_types types;
        types.init_constants();

        return types;
    }

    void throw_unrecognized_typenum_error(int typenum)
    {
        throw std::runtime_error("Unrecogized typenum " +
                                 std::to_string(typenum) + " encountered.");
    }
};

} // namespace detail

} // namespace tensor
} // namespace dpctl
