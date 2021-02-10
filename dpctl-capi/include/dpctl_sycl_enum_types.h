//===- dpctl_sycl_enum_types.h - C API enums for few sycl enum   -*-C++-*- ===//
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
/// This header defines dpCtl specficif enum types that wrap corresponding Sycl
/// enum classes. These enums are defined primarily so that Python extensions
/// that use DPCTL do not have to include Sycl headers directly.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/ExternC.h"

DPCTL_C_EXTERN_C_BEGIN

/*!
 * @brief Redefinition of DPC++-specific Sycl backend types.
 *
 */
enum DPCTLSyclBackendType
{
    // clang-format off
    DPCTL_UNKNOWN_BACKEND = 0x0,
    DPCTL_OPENCL          = 1 << 16,
    DPCTL_HOST            = 1 << 15,
    DPCTL_LEVEL_ZERO      = 1 << 14,
    DPCTL_CUDA            = 1 << 13
    // clang-format on
};

/*!
 * @brief DPCTL device types that are equivalent to Sycl's device_type.
 *
 */
enum DPCTLSyclDeviceType
{
    // clang-format off
    DPCTL_CPU         = 1 << 0,
    DPCTL_GPU         = 1 << 1,
    DPCTL_ACCELERATOR = 1 << 2,
    DPCTL_CUSTOM      = 1 << 3,
    DPCTL_AUTOMATIC   = 1 << 4,
    DPCTL_HOST_DEVICE = 1 << 5,
    DPCTL_ALL         = 1 << 6
    // IMP: before adding new values here look at DPCTLSyclBackendType enum. The
    // values should not overlap.
    // clang-format on
};

/*!
 * @brief Supported types for kernel arguments to be passed to a Sycl kernel
 * using DPCTL.
 *
 * \todo Add support for sycl::buffer
 *
 */
typedef enum
{
    DPCTL_CHAR,
    DPCTL_SIGNED_CHAR,
    DPCTL_UNSIGNED_CHAR,
    DPCTL_SHORT,
    DPCTL_INT,
    DPCTL_UNSIGNED_INT,
    DPCTL_UNSIGNED_INT8,
    DPCTL_LONG,
    DPCTL_UNSIGNED_LONG,
    DPCTL_LONG_LONG,
    DPCTL_UNSIGNED_LONG_LONG,
    DPCTL_SIZE_T,
    DPCTL_FLOAT,
    DPCTL_DOUBLE,
    DPCTL_LONG_DOUBLE,
    DPCTL_VOID_PTR
} DPCTLKernelArgType;

DPCTL_C_EXTERN_C_END
