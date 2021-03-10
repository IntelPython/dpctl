//===----- dpctl_sycl_queue_interface.h - C API for sycl::queue  -*-C++-*- ===//
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
/// This header declares a C interface to sycl::queue member functions. Note
/// that sycl::queue constructors are not exposed in this interface. Instead,
/// users should use the functions in dpctl_sycl_queue_manager.h.
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

/*!
 * @brief Delete the pointer after casting it to sycl::queue.
 *
 * @param    QRef           A DPCTLSyclQueueRef pointer that gets deleted.
 */
DPCTL_API
void DPCTLQueue_Delete(__dpctl_take DPCTLSyclQueueRef QRef);

/*!
 * @brief Checks if two DPCTLSyclQueueRef objects point to the same sycl::queue.
 *
 * @param    QRef1          First opaque pointer to the sycl queue.
 * @param    QRef2          Second opaque pointer to the sycl queue.
 * @return   True if the underlying sycl::queue are same, false otherwise.
 */
DPCTL_API
bool DPCTLQueue_AreEq(__dpctl_keep const DPCTLSyclQueueRef QRef1,
                      __dpctl_keep const DPCTLSyclQueueRef QRef2);

/*!
 * @brief Returns the Sycl backend for the provided sycl::queue.
 *
 * @param    QRef           An opaque pointer to the sycl queue.
 * @return   A enum DPCTLSyclBackendType corresponding to the backed for the
 * queue.
 */
DPCTL_API
DPCTLSyclBackendType DPCTLQueue_GetBackend(__dpctl_keep DPCTLSyclQueueRef QRef);

/*!
 * @brief Returns the Sycl context for the queue.
 *
 * @param    QRef           An opaque pointer to the sycl queue.
 * @return   A DPCTLSyclContextRef pointer to the sycl context for the queue.
 */
DPCTL_API
__dpctl_give DPCTLSyclContextRef
DPCTLQueue_GetContext(__dpctl_keep const DPCTLSyclQueueRef QRef);

/*!
 * @brief returns the Sycl device for the queue.
 *
 * @param    QRef           An opaque pointer to the sycl queue.
 * @return   A DPCTLSyclDeviceRef pointer to the sycl device for the queue.
 */
DPCTL_API
__dpctl_give DPCTLSyclDeviceRef
DPCTLQueue_GetDevice(__dpctl_keep const DPCTLSyclQueueRef QRef);

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
 * @param    ArgTypes       An array of DPCTLKernelArgType enum values that
 *                          represent the type of each kernel argument.
 * @param    NArgs          Size of Args and ArgTypes.
 * @param    Range          Defines the overall dimension of the dispatch for
 *                          the kernel. The array can have up to three
 *                          dimensions.
 * @param    NRange         Size of the gRange array.
 * @param    DepEvents      List of dependent DPCTLSyclEventRef objects (events)
 *                          for the kernel. We call sycl::handler.depends_on for
 *                          each of the provided events.
 * @param    NDepEvents     Size of the DepEvents list.
 * @return   An opaque pointer to the sycl::event returned by the
 *           sycl::queue.submit() function.
 */
DPCTL_API
DPCTLSyclEventRef
DPCTLQueue_SubmitRange(__dpctl_keep const DPCTLSyclKernelRef KRef,
                       __dpctl_keep const DPCTLSyclQueueRef QRef,
                       __dpctl_keep void **Args,
                       __dpctl_keep const DPCTLKernelArgType *ArgTypes,
                       size_t NArgs,
                       __dpctl_keep const size_t Range[3],
                       size_t NRange,
                       __dpctl_keep const DPCTLSyclEventRef *DepEvents,
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
 * @param    ArgTypes       An array of DPCTLKernelArgType enum values that
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
 * @param    DepEvents      List of dependent DPCTLSyclEventRef objects (events)
 *                          for the kernel. We call sycl::handler.depends_on for
 *                          each of the provided events.
 * @param    NDepEvents     Size of the DepEvents list.
 * @return   An opaque pointer to the sycl::event returned by the
 *           sycl::queue.submit() function.
 */
DPCTL_API
DPCTLSyclEventRef
DPCTLQueue_SubmitNDRange(__dpctl_keep const DPCTLSyclKernelRef KRef,
                         __dpctl_keep const DPCTLSyclQueueRef QRef,
                         __dpctl_keep void **Args,
                         __dpctl_keep const DPCTLKernelArgType *ArgTypes,
                         size_t NArgs,
                         __dpctl_keep const size_t gRange[3],
                         __dpctl_keep const size_t lRange[3],
                         size_t NDims,
                         __dpctl_keep const DPCTLSyclEventRef *DepEvents,
                         size_t NDepEvents);

/*!
 * @brief Calls the sycl::queue.submit function to do a blocking wait on all
 * enqueued tasks in the queue.
 *
 * @param    QRef           Opaque pointer to a sycl::queue.
 */
DPCTL_API
void DPCTLQueue_Wait(__dpctl_keep const DPCTLSyclQueueRef QRef);

/*!
 * @brief C-API wrapper for sycl::queue::memcpy, the function waits on an event
 * till the memcpy operation completes.
 *
 * @param    QRef           An opaque pointer to the sycl queue.
 * @param    Dest           An USM pointer to the destination memory.
 * @param    Src            An USM pointer to the source memory.
 * @param    Count          A number of bytes to copy.
 */
DPCTL_API
void DPCTLQueue_Memcpy(__dpctl_keep const DPCTLSyclQueueRef QRef,
                       void *Dest,
                       const void *Src,
                       size_t Count);

/*!
 * @brief C-API wrapper for sycl::queue::prefetch, the function waits on an
 * event till the prefetch operation completes.
 *
 * @param    QRef           An opaque pointer to the sycl queue.
 * @param    Ptr            An USM pointer to memory.
 * @param    Count          A number of bytes to prefetch.
 */
DPCTL_API
void DPCTLQueue_Prefetch(__dpctl_keep DPCTLSyclQueueRef QRef,
                         const void *Ptr,
                         size_t Count);

/*!
 * @brief C-API wrapper for sycl::queue::mem_advise, the function waits on an
 * event till the operation completes.
 *
 * @param    QRef           An opaque pointer to the sycl queue.
 * @param    Ptr            An USM pointer to memory.
 * @param    Count          A number of bytes to prefetch.
 * @param    Advice         Device-defined advice for the specified allocation.
 *                          A value of 0 reverts the advice for Ptr to the
 *                          default behavior.
 */
DPCTL_API
void DPCTLQueue_MemAdvise(__dpctl_keep DPCTLSyclQueueRef QRef,
                          const void *Ptr,
                          size_t Count,
                          int Advice);

DPCTL_C_EXTERN_C_END
