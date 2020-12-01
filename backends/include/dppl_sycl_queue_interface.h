//===----------- dppl_sycl_queue_interface.h - dpctl-C_API ---*---C++ -*---===//
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
/// This header declares a C interface to sycl::queue member functions. Note
/// that sycl::queue constructors are not exposed in this interface. Instead,
/// users should use the functions in dppl_sycl_queue_manager.h.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "dppl_data_types.h"
#include "dppl_sycl_enum_types.h"
#include "dppl_sycl_types.h"
#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"

DPPL_C_EXTERN_C_BEGIN

/*!
 * @brief Delete the pointer after casting it to sycl::queue.
 *
 * @param    QRef           A DPPLSyclQueueRef pointer that gets deleted.
 */
DPPL_API
void DPPLQueue_Delete (__dppl_take DPPLSyclQueueRef QRef);

/*!
 * @brief Checks if two DPPLSyclQueueRef objects point to the same sycl::queue.
 *
 * @param    QRef1          First opaque pointer to the sycl queue.
 * @param    QRef2          Second opaque pointer to the sycl queue.
 * @return   True if the underlying sycl::queue are same, false otherwise.
 */
DPPL_API
bool DPPLQueue_AreEq (__dppl_keep const DPPLSyclQueueRef QRef1,
                      __dppl_keep const DPPLSyclQueueRef QRef2);

/*!
 * @brief Returns the Sycl backend for the provided sycl::queue.
 *
 * @param    QRef           An opaque pointer to the sycl queue.
 * @return   A enum DPPLSyclBackendType corresponding to the backed for the
 * queue.
 */
DPPL_API
DPPLSyclBackendType DPPLQueue_GetBackend (__dppl_keep DPPLSyclQueueRef QRef);

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
 * @brief Submits the kernel to the specified queue with the provided range
 * argument.
 *
 * A wrapper over sycl::queue.submit(). The function takes an interoperability
 * kernel, the kernel arguments, and a Sycl queue as input. The kernel is
 * submitted as parallel_for(range<NRange>, *unwrap(KRef)).
 *
 * \todo sycl::buffer arguments are not supported yet.
 * \todo Add support for id<Dims> WorkItemOffset
 *
 * @param    KRef           Opaque pointer to an OpenCL interoperability kernel
 *                          wrapped inside a sycl::kernel.
 * @param    QRef           Opaque pointer to the sycl::queue where the kernel
 *                          will be enqueued.
 * @param    Args           An array of void* pointers that represent the
 *                          kernel arguments for the kernel.
 * @param    ArgTypes       An array of DPPLKernelArgType enum values that
 *                          represent the type of each kernel argument.
 * @param    NArgs          Size of Args and ArgTypes.
 * @param    Range          Defines the overall dimension of the dispatch for
 *                          the kernel. The array can have up to three
 *                          dimensions.
 * @param    NRange         Size of the gRange array.
 * @param    DepEvents      List of dependent DPPLSyclEventRef objects (events)
 *                          for the kernel. We call sycl::handler.depends_on for
 *                          each of the provided events.
 * @param    NDepEvents     Size of the DepEvents list.
 * @return   An opaque pointer to the sycl::event returned by the
 *           sycl::queue.submit() function.
 */
DPPL_API
DPPLSyclEventRef
DPPLQueue_SubmitRange (__dppl_keep const DPPLSyclKernelRef KRef,
                       __dppl_keep const DPPLSyclQueueRef QRef,
                       __dppl_keep void **Args,
                       __dppl_keep const DPPLKernelArgType *ArgTypes,
                       size_t NArgs,
                       __dppl_keep const size_t Range[3],
                       size_t NRange,
                       __dppl_keep const DPPLSyclEventRef *DepEvents,
                       size_t NDepEvents);

/*!
 * @brief Submits the kernel to the specified queue with the provided nd_range
 * argument.
 *
 * A wrapper over sycl::queue.submit(). The function takes an interoperability
 * kernel, the kernel arguments, and a Sycl queue as input. The kernel is
 * submitted as parallel_for(nd_range<NRange>, *unwrap(KRef)).
 *
 * \todo sycl::buffer arguments are not supported yet.
 * \todo Add support for id<Dims> WorkItemOffset
 *
 * @param    KRef           Opaque pointer to an OpenCL interoperability kernel
 *                          wrapped inside a sycl::kernel.
 * @param    QRef           Opaque pointer to the sycl::queue where the kernel
 *                          will be enqueued.
 * @param    Args           An array of void* pointers that represent the
 *                          kernel arguments for the kernel.
 * @param    ArgTypes       An array of DPPLKernelArgType enum values that
 *                          represent the type of each kernel argument.
 * @param    NArgs          Size of Args.
 * @param    gRange         Defines the overall dimension of the dispatch for
 *                          the kernel. The array can have up to three
 *                          dimensions.
 * @param    lRange         Defines the iteration domain of a single work-group
 *                          in a parallel dispatch. The array can have up to
 *                          three dimensions.
 * @param    NDims          The number of dimensions for both local and global
 *                          ranges.
 * @param    DepEvents      List of dependent DPPLSyclEventRef objects (events)
 *                          for the kernel. We call sycl::handler.depends_on for
 *                          each of the provided events.
 * @param    NDepEvents     Size of the DepEvents list.
 * @return   An opaque pointer to the sycl::event returned by the
 *           sycl::queue.submit() function.
 */
DPPL_API
DPPLSyclEventRef
DPPLQueue_SubmitNDRange(__dppl_keep const DPPLSyclKernelRef KRef,
                        __dppl_keep const DPPLSyclQueueRef QRef,
                        __dppl_keep void **Args,
                        __dppl_keep const DPPLKernelArgType *ArgTypes,
                        size_t NArgs,
                        __dppl_keep const size_t gRange[3],
                        __dppl_keep const size_t lRange[3],
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
 * @brief C-API wrapper for sycl::queue::memcpy, the function waits on an event
 * till the memcpy operation completes.
 *
 * @param    QRef           An opaque pointer to the sycl queue.
 * @param    Dest           An USM pointer to the destination memory.
 * @param    Src            An USM pointer to the source memory.
 * @param    Count          A number of bytes to copy.
 */
DPPL_API
void DPPLQueue_Memcpy (__dppl_keep const DPPLSyclQueueRef QRef,
                       void *Dest, const void *Src, size_t Count);

/*!
 * @brief C-API wrapper for sycl::queue::prefetch, the function waits on an event
 * till the prefetch operation completes.
 *
 * @param    QRef           An opaque pointer to the sycl queue.
 * @param    Ptr            An USM pointer to memory.
 * @param    Count          A number of bytes to prefetch.
 */
DPPL_API
void DPPLQueue_Prefetch (__dppl_keep DPPLSyclQueueRef QRef,
                         const void *Ptr, size_t Count);

/*!
 * @brief C-API wrapper for sycl::queue::mem_advise, the function waits on an event
 * till the operation completes.
 *
 * @param    QRef           An opaque pointer to the sycl queue.
 * @param    Ptr            An USM pointer to memory.
 * @param    Count          A number of bytes to prefetch.
 * @param    Advice         Device-defined advice for the specified allocation. 
 *                           A value of 0 reverts the advice for Ptr to the default behavior.
 */
DPPL_API
void DPPLQueue_MemAdvise (__dppl_keep DPPLSyclQueueRef QRef,
                          const void *Ptr, size_t Count, int Advice);

DPPL_C_EXTERN_C_END
