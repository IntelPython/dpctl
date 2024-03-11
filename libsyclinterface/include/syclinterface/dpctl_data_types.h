//===-------------- dpctl_data_types.h - Defines integer types   -*-C++-*- ===//
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
/// This file was copied over from the LLVM-C (include/llvm-c/DataTypes.h).
/// This file contains definitions to figure out the size of _HOST_ data types.
/// This file is important because different host OS's define different macros,
/// which makes portability tough.  This file exports the following
/// definitions:
///
/// [u]int(32|64)_t : typedefs for signed and unsigned 32/64 bit system types
/// [U]INT(8|16|32|64)_(MIN|MAX) : Constants for the min and max values.
///
/// No library is required when using these functions.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <inttypes.h>
#include <stdint.h>

#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifndef _MSC_VER

#if !defined(UINT32_MAX)
#error "The standard header <cstdint> is not C++11 compliant. Must #define "\
        "__STDC_LIMIT_MACROS before #including llvm-c/DataTypes.h"
#endif

#if !defined(UINT32_C)
#error "The standard header <cstdint> is not C++11 compliant. Must #define "\
        "__STDC_CONSTANT_MACROS before #including llvm-c/DataTypes.h"
#endif

/* Note that <inttypes.h> includes <stdint.h>, if this is a C99 system. */
#include <sys/types.h>

#ifdef _AIX
// GCC is strict about defining large constants: they must have LL modifier.
#undef INT64_MAX
#undef INT64_MIN
#endif

#else /* _MSC_VER */
#ifdef __cplusplus
#include <cstddef>
#include <cstdlib>
#else
#include <stddef.h>
#include <stdlib.h>
#endif
#include <sys/types.h>

#if defined(_WIN64)
typedef signed __int64 ssize_t;
#else
typedef signed int ssize_t;
#endif /* _WIN64 */

#endif /* _MSC_VER */

/*!
    @brief Represents the largest possible value of a 64 bit signed integer.
*/
#if !defined(INT64_MAX)
#define INT64_MAX 9223372036854775807LL
#endif

/*!
    @brief Represents the smallest possible value of a 64 bit signed integer.
*/
#if !defined(INT64_MIN)
#define INT64_MIN ((-INT64_MAX) - 1)
#endif

/*!
    @brief Represents the largest possible value of a 64bit unsigned integer.
*/
#if !defined(UINT64_MAX)
#define UINT64_MAX 0xffffffffffffffffULL
#endif

/*!
    @brief Represents a positive expression of type float.
*/
#ifndef HUGE_VALF
#define HUGE_VALF (float)HUGE_VAL
#endif
