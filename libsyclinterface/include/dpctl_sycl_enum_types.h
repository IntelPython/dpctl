//===- dpctl_sycl_enum_types.h - C API enums for few sycl enum   -*-C++-*- ===//
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
/// This header defines dpctl specific enum types that wrap corresponding Sycl
/// enum classes. These enums are defined primarily so that Python extensions
/// that use DPCTL do not have to include Sycl headers directly.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/ExternC.h"

DPCTL_C_EXTERN_C_BEGIN

/*!
 * @brief Enum types for SYCL's USM allocator types.
 *
 */
typedef enum
{
    DPCTL_USM_UNKNOWN = 0,
    DPCTL_USM_DEVICE,
    DPCTL_USM_SHARED,
    DPCTL_USM_HOST
} DPCTLSyclUSMType;

/*!
 * @brief Redefinition of DPC++-specific Sycl backend types.
 *
 */
typedef enum
{
    // clang-format off
    DPCTL_CUDA            = 1 << 16,
    DPCTL_LEVEL_ZERO      = 1 << 17,
    DPCTL_OPENCL          = 1 << 18,
    DPCTL_UNKNOWN_BACKEND = 0,
    DPCTL_ALL_BACKENDS    = ((1<<5)-1) << 16
    // clang-format on
} DPCTLSyclBackendType;

/*!
 * @brief DPCTL device types that are equivalent to Sycl's device_type.
 *
 */
typedef enum
{
    // Note: before adding new values here look at DPCTLSyclBackendType enum.
    // The values should not overlap.

    // clang-format off
    DPCTL_ACCELERATOR    = 1 << 0,
    DPCTL_AUTOMATIC      = 1 << 1,
    DPCTL_CPU            = 1 << 2,
    DPCTL_CUSTOM         = 1 << 3,
    DPCTL_GPU            = 1 << 4,
    DPCTL_ALL            = (1 << 6) - 1,
    DPCTL_UNKNOWN_DEVICE = 0
    // clang-format on
} DPCTLSyclDeviceType;

/*!
 * @brief Supported types for kernel arguments to be passed to a Sycl kernel
 * using DPCTL.
 *
 * \todo Add support for sycl::buffer
 *
 */
typedef enum
{
    DPCTL_INT8_T,
    DPCTL_UINT8_T,
    DPCTL_INT16_T,
    DPCTL_UINT16_T,
    DPCTL_INT32_T,
    DPCTL_UINT32_T,
    DPCTL_INT64_T,
    DPCTL_UINT64_T,
    DPCTL_FLOAT32_T,
    DPCTL_FLOAT64_T,
    DPCTL_VOID_PTR,
    DPCTL_LOCAL_ACCESSOR,
    DPCTL_UNSUPPORTED_KERNEL_ARG
} DPCTLKernelArgType;

/*!
 * @brief DPCTL device has an associated set of aspects which identify
 * characteristics of the device.
 *
 */
typedef enum
{
    host,
    cpu,
    gpu,
    accelerator,
    custom,
    fp16,
    fp64,
    atomic64,
    image,
    online_compiler,
    online_linker,
    queue_profiling,
    usm_device_allocations,
    usm_host_allocations,
    usm_shared_allocations,
    usm_system_allocations,
    usm_atomic_host_allocations,
    usm_atomic_shared_allocations,
    host_debuggable,
} DPCTLSyclAspectType;

/*!
 * @brief DPCTL analogue of ``sycl::info::partition_affinity_domain`` enum.
 *
 */
typedef enum
{
    not_applicable,
    numa,
    L4_cache,
    L3_cache,
    L2_cache,
    L1_cache,
    next_partitionable
} DPCTLPartitionAffinityDomainType;

/*!
 * @brief Enums to depict the properties that can be passed to a sycl::queue
 * constructor.
 *
 */
typedef enum
{
    // clang-format off
    DPCTL_DEFAULT_PROPERTY = 0,
    DPCTL_ENABLE_PROFILING = 1 << 0,
    DPCTL_IN_ORDER         = 1 << 1
    // clang-format on
} DPCTLQueuePropertyType;

typedef enum
{
    DPCTL_UNKNOWN_STATUS,
    DPCTL_SUBMITTED,
    DPCTL_RUNNING,
    DPCTL_COMPLETE
} DPCTLSyclEventStatusType;

typedef enum
{
    DPCTL_MEM_CACHE_TYPE_INDETERMINATE,
    DPCTL_MEM_CACHE_TYPE_NONE,
    DPCTL_MEM_CACHE_TYPE_READ_ONLY,
    DPCTL_MEM_CACHE_TYPE_READ_WRITE
} DPCTLGlobalMemCacheType;

DPCTL_C_EXTERN_C_END
