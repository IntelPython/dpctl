//===--- dppl_sycl_queue_interface.h - DPPL-SYCL interface ---*---C++ -*---===//
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
/// This header declares a C interface to sycl::queue member functions. Note
/// that sycl::queue constructors are not exposed in this interface. Instead,
/// users should use the functions in dppl_sycl_queue_manager.h.
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
 * @brief Supported types for kernel arguments to be passed to a Sycl kernel.
 *
 * \todo Add support for sycl::buffer
 *
 */
typedef enum
{
    DPPL_CHAR,
    DPPL_SIGNED_CHAR,
    DPPL_UNSIGNED_CHAR,
    DPPL_SHORT,
    DPPL_INT,
    DPPL_UNSIGNED_INT,
    DPPL_LONG,
    DPPL_UNSIGNED_LONG,
    DPPL_LONG_LONG,
    DPPL_UNSIGNED_LONG_LONG,
    DPPL_SIZE_T,
    DPPL_FLOAT,
    DPPL_DOUBLE,
    DPPL_LONG_DOUBLE,
    DPPL_VOID_PTR
} DPPLKernelArgType;

/*!
 * @brief Delete the pointer after casting it to sycl::queue.
 *
 * @param    QRef           A DPPLSyclQueueRef pointer that gets deleted.
 */
DPPL_API
void DPPLQueue_Delete (__dppl_take DPPLSyclQueueRef QRef);

/*!
 * @brief Returns the Sycl context for the queue.
 *
 * @param    QRef           An opaque pointer to the sycl queue.
 * @return   A DPPLSyclContextRef pointer to the sycl context for the queue.
 */
DPPL_API
__dppl_give DPPLSyclContextRef
DPPLQueue_GetContext (__dppl_keep const DPPLSyclQueueRef QRef);

/*!
 * @brief returns the Sycl device for the queue.
 *
 * @param    QRef           An opaque pointer to the sycl queue.
 * @return   A DPPLSyclDeviceRef pointer to the sycl device for the queue.
 */
DPPL_API
__dppl_give DPPLSyclDeviceRef
DPPLQueue_GetDevice (__dppl_keep const DPPLSyclQueueRef QRef);

/*!
 * @brief Submits the kernel to the specified queue using give arguments.
 *
 * A wrapper over sycl::queue.submit(). The function takes an OpenCL
 * interoperability kernel, the kernel arguments, and a sycl queue as input
 * arguments. The kernel arguments are passed in as an array of the
 * DPPLKernelArg tagged union.
 *
 * \todo sycl::buffer arguments are not supported yet.
 * \todo Add support for id<Dims> WorkItemOffset
 *
 * @param    KRef           Opaque pointer to a OpenCL interoperability kernel
 *                          wrapped inside a sycl::kernel.
 * @param    QRef           Opaque pointer to the sycl::queue where the kernel
 *                          will be enqueued.
 * @param    Args           An array of the DPPLKernelArg tagged union type that
 *                          represents the kernel arguments for the kernel.
 * @param    NArgs          The number of kernel arguments (size of Args array).
 * @param    Range          Array storing the range dimensions that can have a
 *                          maximum size of three. Note the number of values
 *                          in the array depends on the number of dimensions.
 * @param    NDims          Number of dimensions in the range (size of Range).
 * @param    DepEvents      List of dependent DPPLSyclEventRef objects (events)
 *                          for the kernel. We call sycl::handler.depends_on for
 *                          each of the provided events.
 * @param    NDepEvents     Size of the DepEvents list.
 * @return   A opaque pointer to the sycl::event returned by the
 *           sycl::queue.submit() function.
 */
DPPL_API
DPPLSyclEventRef
DPPLQueue_SubmitRange (__dppl_keep const DPPLSyclKernelRef KRef,
                       __dppl_keep const DPPLSyclQueueRef QRef,
                       __dppl_keep void **Args,
                       __dppl_keep const DPPLKernelArgType *ArgTypes,
                       size_t NArgs,
                       size_t const Range[3],
                       size_t NDims,
                       __dppl_keep const DPPLSyclEventRef *DepEvents,
                       size_t NDepEvents);

/*!
 * @brief Calls the sycl::queue.submit function to do a blocking wait on all
 * enqueued tasks in the queue.
 *
 * @param    QRef           Opaque pointer to a sycl::queue.
 */
DPPL_API
void
DPPLQueue_Wait (__dppl_keep const DPPLSyclQueueRef QRef);

/*!
 * @brief C-API wrapper for sycl::queue::memcpy. It waits an event.
 *
 * @param    QRef           A DPPLSyclQueueRef pointer that gets deleted.
 */
DPPL_API
void DPPLQueue_memcpy (__dppl_keep const DPPLSyclQueueRef QRef,
                       void *Dest, const void *Src, size_t Count);

DPPL_C_EXTERN_C_END
