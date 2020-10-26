//===------------- dppl_sycl_usm_interface.h - dpctl-C_API ---*---C++ -*---===//
//
//               Data Parallel Control Library (dpCtl)
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
/// This header declares a C interface to sycl::usm interface functions.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "dppl_data_types.h"
#include "dppl_sycl_types.h"
#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"

DPPL_C_EXTERN_C_BEGIN

/*!
 * @brief Create USM shared memory.
 *
 * @param    size     Number of bytes to allocate
 * @param    QRef     Sycl queue reference to use in allocation
 *
 * @return The pointer to USM shared memory. On failure, returns nullptr.
 */
DPPL_API
__dppl_give DPPLSyclUSMRef
DPPLmalloc_shared (size_t size, __dppl_keep const DPPLSyclQueueRef QRef);

/*!
 * @brief Create USM shared memory.
 *
 * @param  alignment   Allocation's byte alignment
 * @param  size        Number of bytes to allocate
 * @param  QRef        Sycl queue reference to use in allocation
 *
 * @return The pointer to USM shared memory with the requested alignment.
 * On failure, returns nullptr.
 */
DPPL_API
__dppl_give DPPLSyclUSMRef
DPPLaligned_alloc_shared (size_t alignment, size_t size,
                          __dppl_keep const DPPLSyclQueueRef QRef);

/*!
 * @brief Create USM host memory.
 *
 * @param    size     Number of bytes to allocate
 * @param    QRef     Sycl queue reference to use in allocation
 *
 * @return The pointer to USM host memory. On failure, returns nullptr.
 */
DPPL_API
__dppl_give DPPLSyclUSMRef
DPPLmalloc_host (size_t size, __dppl_keep const DPPLSyclQueueRef QRef);

/*!
 * @brief Create USM host memory.
 *
 * @param  alignment   Allocation's byte alignment
 * @param  size        Number of bytes to allocate
 * @param  QRef        Sycl queue reference to use in allocation
 *
 * @return The pointer to USM host memory with the requested alignment.
 * On failure, returns nullptr.
 */
DPPL_API
__dppl_give DPPLSyclUSMRef
DPPLaligned_alloc_host (size_t alignment, size_t size,
                        __dppl_keep const DPPLSyclQueueRef QRef);

/*!
 * @brief Create USM device memory.
 *
 * @param    size     Number of bytes to allocate
 * @param    QRef     Sycl queue reference to use in allocation
 *
 * @return The pointer to USM device memory. On failure, returns nullptr.
 */
DPPL_API
__dppl_give DPPLSyclUSMRef
DPPLmalloc_device (size_t size, __dppl_keep const DPPLSyclQueueRef QRef);

/*!
 * @brief Create USM device memory.
 *
 * @param    alignment   Allocation's byte alignment
 * @param    size        Number of bytes to allocate
 * @param    QRef        Sycl queue reference to use in allocation
 *
 * @return The pointer to USM device memory with requested alignment.
 * On failure, returns nullptr.
 */
DPPL_API
__dppl_give DPPLSyclUSMRef
DPPLaligned_alloc_device (size_t alignment, size_t size,
                          __dppl_keep const DPPLSyclQueueRef QRef);

/*!
 * @brief Free USM memory.
 *
 * @param   MRef      USM pointer to free
 * @param   QRef      Sycl queue reference to use.
 *
 * USM pointer must have been allocated using the same context as the one
 * used to construct the queue.
 */
DPPL_API
void DPPLfree_with_queue (__dppl_take DPPLSyclUSMRef MRef,
                          __dppl_keep const DPPLSyclQueueRef QRef);

/*!
 * @brief Free USM memory.
 *
 */
DPPL_API
void DPPLfree_with_context (__dppl_take DPPLSyclUSMRef MRef,
                            __dppl_keep const DPPLSyclContextRef CRef);

/*!
 * @brief Get pointer type.
 *
 * @param    MRef      USM Memory 
 * @param    CRef 
 *
 * @return "host", "device", "shared" or "unknown"
 */
DPPL_API
const char *
DPPLUSM_GetPointerType (__dppl_keep const DPPLSyclUSMRef MRef,
                        __dppl_keep const DPPLSyclContextRef CRef);

/*!
 * @brief Get the device associated with USM pointer.
 *
 * @param  MRef    USM pointer
 * @param  CRef    Sycl context reference associated with the pointer
 *
 * @return A DPPLSyclDeviceRef pointer to the sycl device.
 */
DPPL_API
DPPLSyclDeviceRef
DPPLUSM_GetPointerDevice (__dppl_keep const DPPLSyclUSMRef MRef,
                          __dppl_keep const DPPLSyclContextRef CRef);
DPPL_C_EXTERN_C_END
