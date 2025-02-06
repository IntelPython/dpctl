//=== sycl_complex.hpp ----------------------------------------  *-C++-*--/===//
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
/// This file defines a macro for defining the SYCL_EXT_ONEAPI_COMPLEX macro
/// and indirect inclusion of the experimental oneAPI SYCL complex extension
/// header file.
//===---------------------------------------------------------------------===//

#pragma once

#define SYCL_EXT_ONEAPI_COMPLEX
#if __has_include(<sycl/ext/oneapi/experimental/sycl_complex.hpp>)
#include <sycl/ext/oneapi/experimental/sycl_complex.hpp>
#else
#include <sycl/ext/oneapi/experimental/complex/complex.hpp>
#endif

namespace exprm_ns = sycl::ext::oneapi::experimental;
