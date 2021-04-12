//===- dpctl_sycl_enum_types.h - C API enums for few sycl enum   -*-C++-*- ===//
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
/// This header defines dpctl specific enum types that wrap corresponding Sycl
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
    DPCTL_CUDA            = 1 << 13,
    DPCTL_HOST            = 1 << 14,
    DPCTL_LEVEL_ZERO      = 1 << 15,
    DPCTL_OPENCL          = 1 << 16,
    DPCTL_UNKNOWN_BACKEND = 0,
    DPCTL_ALL_BACKENDS    = ((1<<10)-1) << 7
    // clang-format on
};

/*!
 * @brief DPCTL device types that are equivalent to Sycl's device_type.
 *
 */
enum DPCTLSyclDeviceType
{
    // Note: before adding new values here look at DPCTLSyclBackendType enum.
    // The values should not overlap.

    // clang-format off
    DPCTL_ACCELERATOR    = 1 << 1,
    DPCTL_AUTOMATIC      = 1 << 2,
    DPCTL_CPU            = 1 << 3,
    DPCTL_CUSTOM         = 1 << 4,
    DPCTL_GPU            = 1 << 5,
    DPCTL_HOST_DEVICE    = 1 << 6,
    DPCTL_ALL            = (1 << 7) -1 ,
    DPCTL_UNKNOWN_DEVICE = 0
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

/*!
 * @brief DPCTL device has an associated set of aspects which identify
 * characteristics of the device.
 *
 */
enum DPCTLSyclAspectType
{
    host,
    cpu,
    gpu,
    accelerator,
    custom,
    fp16,
    fp64,
    int64_base_atomics,
    int64_extended_atomics,
    image,
    online_compiler,
    online_linker,
    queue_profiling,
    usm_device_allocations,
    usm_host_allocations,
    usm_shared_allocations,
    usm_restricted_shared_allocations,
    usm_system_allocator
};

/*!
 * @brief DPCTL analogue of ``sycl::info::partition_affinity_domain`` enum.
 *
 */
enum DPCTLPartitionAffinityDomainType
{
    not_applicable,
    numa,
    L4_cache,
    L3_cache,
    L2_cache,
    L1_cache,
    next_partitionable
};

/*!
 * @brief Enums to depict the properties that can be passed to a sycl::queue
 * constructor.
 *
 */
typedef enum
{
    // clang-format off
    DPCTL_DEFAULT_PROPERTY = 0,
    DPCTL_ENABLE_PROFILING = 1 << 1,
    DPCTL_IN_ORDER         = 1 << 2
    // clang-format on
} DPCTLQueuePropertyType;

DPCTL_C_EXTERN_C_END
