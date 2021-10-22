//===--------- dpctl_elementary_macros.h - Defines helper macros -*-C++-*- ===//
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
/// Several helper C preprocessor macros used elsewhere in our code are defined
/// here.
///
//===----------------------------------------------------------------------===//

#pragma once

/*!
 * @brief defines a "__declspec(dllexport)" wrapper for windows.
 *
 */
#ifdef _WIN32
#ifdef DPCTLSyclInterface_EXPORTS
#define DPCTL_API __declspec(dllexport)
#else
#define DPCTL_API __declspec(dllimport)
#endif
#else
#define DPCTL_API
#endif

/*!
 * @brief Convenience macros to add "extern C {" statements to headers.
 *
 */
#ifdef __cplusplus
#define DPCTL_C_EXTERN_C_BEGIN                                                 \
    extern "C"                                                                 \
    {
#define DPCTL_C_EXTERN_C_END }
#else
#define DPCTL_C_EXTERN_C_BEGIN
#define DPCTL_C_EXTERN_C_END
#endif

/*!
 * @brief Convenience macro to add deprecation attributes to functions.
 *
 */
#define DEPRACATION_NOTICE(notice, FN) notice " " #FN

#if defined(__GNUC__) && !defined(__clang__)
#define DPCTL_DEPRECATE(msg, repl_fn_name)                                     \
    __attribute__((deprecated(DEPRACATION_NOTICE(msg, Use repl_fn_name))))
#elif defined(__clang__)
#define DPCTL_DEPRECATE(msg, repl_fn_name)                                     \
    __attribute__((deprecated(msg, repl_fn_name)))
#endif
