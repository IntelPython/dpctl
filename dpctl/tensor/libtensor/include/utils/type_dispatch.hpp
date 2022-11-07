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

struct usm_ndarray_types
{

    int typenum_to_lookup_id(int typenum)
    {
        using typenum_t = dpctl::tensor::detail::typenum_t;
        auto const &api = ::dpctl::detail::dpctl_capi::get();

        if (typenum == api.UAR_DOUBLE_) {
            return static_cast<int>(typenum_t::DOUBLE);
        }
        else if (typenum == api.UAR_INT64_) {
            return static_cast<int>(typenum_t::INT64);
        }
        else if (typenum == api.UAR_INT32_) {
            return static_cast<int>(typenum_t::INT32);
        }
        else if (typenum == api.UAR_BOOL_) {
            return static_cast<int>(typenum_t::BOOL);
        }
        else if (typenum == api.UAR_CDOUBLE_) {
            return static_cast<int>(typenum_t::CDOUBLE);
        }
        else if (typenum == api.UAR_FLOAT_) {
            return static_cast<int>(typenum_t::FLOAT);
        }
        else if (typenum == api.UAR_INT16_) {
            return static_cast<int>(typenum_t::INT16);
        }
        else if (typenum == api.UAR_INT8_) {
            return static_cast<int>(typenum_t::INT8);
        }
        else if (typenum == api.UAR_UINT64_) {
            return static_cast<int>(typenum_t::UINT64);
        }
        else if (typenum == api.UAR_UINT32_) {
            return static_cast<int>(typenum_t::UINT32);
        }
        else if (typenum == api.UAR_UINT16_) {
            return static_cast<int>(typenum_t::UINT16);
        }
        else if (typenum == api.UAR_UINT8_) {
            return static_cast<int>(typenum_t::UINT8);
        }
        else if (typenum == api.UAR_CFLOAT_) {
            return static_cast<int>(typenum_t::CFLOAT);
        }
        else if (typenum == api.UAR_HALF_) {
            return static_cast<int>(typenum_t::HALF);
        }
        else if (typenum == api.UAR_INT_ || typenum == api.UAR_UINT_) {
            switch (sizeof(int)) {
            case sizeof(std::int32_t):
                return ((typenum == api.UAR_INT_)
                            ? static_cast<int>(typenum_t::INT32)
                            : static_cast<int>(typenum_t::UINT32));
            case sizeof(std::int64_t):
                return ((typenum == api.UAR_INT_)
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
    void throw_unrecognized_typenum_error(int typenum)
    {
        throw std::runtime_error("Unrecogized typenum " +
                                 std::to_string(typenum) + " encountered.");
    }
};

} // namespace detail

} // namespace tensor
} // namespace dpctl
