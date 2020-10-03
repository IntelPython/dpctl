//===----- Support/CBindingWrapping.h - DPPL-SYCL interface --*-- C ---*---===//
//
//               Python Data Parallel Processing Library (PyDPPL)
//
// Copyright 2020 Intel Corporation
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
/// This file declares the wrapping macros for the DPPL C interface.
///
//===----------------------------------------------------------------------===//

#pragma once

#define DEFINE_SIMPLE_CONVERSION_FUNCTIONS(ty, ref)              \
  inline ty *unwrap(ref P) { return reinterpret_cast<ty *>(P); } \
                                                                 \
  inline ref wrap(const ty *P) {                                 \
    return reinterpret_cast<ref>(const_cast<ty *>(P));           \
  }

#define DEFINE_STDCXX_CONVERSION_FUNCTIONS(ty, ref) \
  DEFINE_SIMPLE_CONVERSION_FUNCTIONS(ty, ref)       \
                                                    \
  template <typename T>                             \
  inline T *unwrap(ref P) {                         \
    T *Q = (T *)unwrap(P);                          \
    assert(Q && "Invalid cast!");                   \
    return Q;                                       \
  }
