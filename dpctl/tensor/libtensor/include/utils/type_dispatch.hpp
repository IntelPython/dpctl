//===--type_dispatch.cpp - Type-dispatch table building utils ----*-C++-*- ===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2021 Intel Corporation
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
        assert(per_dsTy.size() == _num_types);
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

} // namespace detail

} // namespace tensor
} // namespace dpctl
