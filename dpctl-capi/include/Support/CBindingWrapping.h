//===- CBindingWrapping.h - Wrappers for casting C pointers      -*-C++-*- ===//
//
//                      Data Parallel Control (dpCtl)
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
/// This file declares the wrapping macros for the dpCtl C interface.
///
//===----------------------------------------------------------------------===//

#pragma once

/*!
    @brief Creates two convenience functions to reinterpret_cast an opaque
    pointer to a pointer to a Sycl type and vice-versa.
*/
#define DEFINE_SIMPLE_CONVERSION_FUNCTIONS(ty, ref)                            \
    __attribute__((unused)) inline ty *unwrap(ref P)                           \
    {                                                                          \
        return reinterpret_cast<ty *>(P);                                      \
    }                                                                          \
                                                                               \
    __attribute__((unused)) inline ref wrap(const ty *P)                       \
    {                                                                          \
        return reinterpret_cast<ref>(const_cast<ty *>(P));                     \
    }

/*!
    @brief Add an overloaded unwrap to assert that a pointer can be legally
    cast. @see DEFINE_SIMPLE_CONVERSION_FUNCTIONS()
*/
#define DEFINE_STDCXX_CONVERSION_FUNCTIONS(ty, ref)                            \
    DEFINE_SIMPLE_CONVERSION_FUNCTIONS(ty, ref)                                \
                                                                               \
    template <typename T> __attribute__((unused)) inline T *unwrap(ref P)      \
    {                                                                          \
        T *Q = (T *)unwrap(P);                                                 \
        assert(Q && "Invalid cast!");                                          \
        return Q;                                                              \
    }
