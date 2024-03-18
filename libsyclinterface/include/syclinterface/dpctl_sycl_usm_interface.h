//===---- dpctl_sycl_usm_interface.h - C API for USM allocators  -*-C++-*- ===//
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
/// This header declares a C interface to sycl::usm interface functions.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"
#include "dpctl_data_types.h"
#include "dpctl_sycl_enum_types.h"
#include "dpctl_sycl_types.h"

DPCTL_C_EXTERN_C_BEGIN

/**
 * @defgroup USMInterface USM Interface
 */

/*!
 * @brief Create USM shared memory.
 *
 * @param    size     Number of bytes to allocate
 * @param    QRef     Sycl queue reference to use in allocation
 *
 * @return The pointer to USM shared memory. On failure, returns nullptr.
 * @ingroup USMInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclUSMRef
DPCTLmalloc_shared(size_t size, __dpctl_keep const DPCTLSyclQueueRef QRef);

/*!
 * @brief Create USM shared memory.
 *
 * @param  alignment   Allocation's byte alignment
 * @param  size        Number of bytes to allocate
 * @param  QRef        Sycl queue reference to use in allocation
 *
 * @return The pointer to USM shared memory with the requested alignment.
 * On failure, returns nullptr.
 * @ingroup USMInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclUSMRef
DPCTLaligned_alloc_shared(size_t alignment,
                          size_t size,
                          __dpctl_keep const DPCTLSyclQueueRef QRef);

/*!
 * @brief Create USM host memory.
 *
 * @param    size     Number of bytes to allocate
 * @param    QRef     Sycl queue reference to use in allocation
 *
 * @return The pointer to USM host memory. On failure, returns nullptr.
 * @ingroup USMInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclUSMRef
DPCTLmalloc_host(size_t size, __dpctl_keep const DPCTLSyclQueueRef QRef);

/*!
 * @brief Create USM host memory.
 *
 * @param  alignment   Allocation's byte alignment
 * @param  size        Number of bytes to allocate
 * @param  QRef        Sycl queue reference to use in allocation
 *
 * @return The pointer to USM host memory with the requested alignment.
 * On failure, returns nullptr.
 * @ingroup USMInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclUSMRef
DPCTLaligned_alloc_host(size_t alignment,
                        size_t size,
                        __dpctl_keep const DPCTLSyclQueueRef QRef);

/*!
 * @brief Create USM device memory.
 *
 * @param    size     Number of bytes to allocate
 * @param    QRef     Sycl queue reference to use in allocation
 *
 * @return The pointer to USM device memory. On failure, returns nullptr.
 * @ingroup USMInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclUSMRef
DPCTLmalloc_device(size_t size, __dpctl_keep const DPCTLSyclQueueRef QRef);

/*!
 * @brief Create USM device memory.
 *
 * @param    alignment   Allocation's byte alignment
 * @param    size        Number of bytes to allocate
 * @param    QRef        Sycl queue reference to use in allocation
 *
 * @return The pointer to USM device memory with requested alignment.
 * On failure, returns nullptr.
 * @ingroup USMInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclUSMRef
DPCTLaligned_alloc_device(size_t alignment,
                          size_t size,
                          __dpctl_keep const DPCTLSyclQueueRef QRef);

/*!
 * @brief Free USM memory.
 *
 * @param   MRef      USM pointer to free
 * @param   QRef      Sycl queue reference to use.
 *
 * USM pointer must have been allocated using the same context as the one
 * used to construct the queue.
 * @ingroup USMInterface
 */
DPCTL_API
void DPCTLfree_with_queue(__dpctl_take DPCTLSyclUSMRef MRef,
                          __dpctl_keep const DPCTLSyclQueueRef QRef);

/*!
 * @brief Free USM memory.
 * @param   MRef      USM pointer to free
 * @param   CRef      Sycl context reference to use.
 * @ingroup USMInterface
 */
DPCTL_API
void DPCTLfree_with_context(__dpctl_take DPCTLSyclUSMRef MRef,
                            __dpctl_keep const DPCTLSyclContextRef CRef);

/*!
 * @brief Returns the USM allocator type for a pointer.
 *
 * @param    MRef      USM allocated pointer
 * @param    CRef      Sycl context reference associated with the pointer
 *
 * @return DPCTLSyclUSMType enum value indicating if the pointer is of USM type
 *         "shared", "host", or "device".
 * @ingroup USMInterface
 */
DPCTL_API
DPCTLSyclUSMType
DPCTLUSM_GetPointerType(__dpctl_keep const DPCTLSyclUSMRef MRef,
                        __dpctl_keep const DPCTLSyclContextRef CRef);

/*!
 * @brief Get the device associated with USM pointer.
 *
 * @param  MRef    USM pointer
 * @param  CRef    Sycl context reference associated with the pointer
 *
 * @return A DPCTLSyclDeviceRef pointer to the sycl device.
 * @ingroup USMInterface
 */
DPCTL_API
DPCTLSyclDeviceRef
DPCTLUSM_GetPointerDevice(__dpctl_keep const DPCTLSyclUSMRef MRef,
                          __dpctl_keep const DPCTLSyclContextRef CRef);
DPCTL_C_EXTERN_C_END
