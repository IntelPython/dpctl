//===--type_dispatch.cpp - Type-dispatch table building utils ----*-C++-*- ===//
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

#include "dpctl4pybind11.hpp"
#include "type_dispatch_building.hpp"

namespace dpctl
{
namespace tensor
{

namespace type_dispatch
{

struct usm_ndarray_types
{

    int typenum_to_lookup_id(int typenum) const
    {
        using typenum_t = ::dpctl::tensor::type_dispatch::typenum_t;
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
        else if (typenum == api.UAR_LONGLONG_ || typenum == api.UAR_ULONGLONG_)
        {
            switch (sizeof(long long)) {
            case sizeof(std::int64_t):
                return ((typenum == api.UAR_LONGLONG_)
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
    void throw_unrecognized_typenum_error(int typenum) const
    {
        throw std::runtime_error("Unrecognized typenum " +
                                 std::to_string(typenum) + " encountered.");
    }
};

} // namespace type_dispatch

} // namespace tensor
} // namespace dpctl
