//===----- dpctl_sycl_queue_interface.h - C API for sycl::queue  -*-C++-*- ===//
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
/// This header declares a C interface to sycl::queue member functions.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"
#include "dpctl_data_types.h"
#include "dpctl_error_handler_type.h"
#include "dpctl_sycl_enum_types.h"
#include "dpctl_sycl_types.h"

DPCTL_C_EXTERN_C_BEGIN

/**
 * @defgroup QueueInterface Queue class C wrapper
 */

/*!
 * @brief A wrapper for sycl::queue constructor to construct a new queue from
 * the provided context, device, async handler and properties bit flags.
 *
 * @param    CRef           An opaque pointer to a sycl::context.
 * @param    DRef           An opaque pointer to a sycl::device
 * @param    handler        A callback function that will be invoked by the
 *                          async_handler used during queue creation. Can be
 *                          NULL if no async_handler is needed.
 * @param    properties     A combination of bit flags using the values defined
 *                          in the DPCTLQueuePropertyType enum. The bit flags
 *                          are used to create a sycl::property_list that is
 *                          passed to the SYCL queue constructor.
 * @return An opaque DPCTLSyclQueueRef pointer containing the new sycl::queue
 * object. A nullptr is returned if the queue could not be created.
 * @ingroup QueueInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclQueueRef
DPCTLQueue_Create(__dpctl_keep const DPCTLSyclContextRef CRef,
                  __dpctl_keep const DPCTLSyclDeviceRef DRef,
                  error_handler_callback *handler,
                  int properties);

/*!
 * @brief Constructs a ``sycl::queue`` object of the specified SYCL device.
 *
 * Constructs a new SYCL queue for the specified SYCL device. The
 * behavior of this function differs from the following queue constructor:
 *
 * @code
 *    queue(
 *        const device &syclDevice,
 *        const async_handler &asyncHandler,
 *        const property_list &propList = {}
 *    )
 * @endcode
 *
 * Unlike the SYCL queue constructor, we try not to create a new SYCL
 * context for the device and instead look to reuse a previously cached
 * SYCL context for the device (refer dpctl_sycl_device_manager.cpp).
 * DPCTL caches contexts only for root devices and for all custom devices the
 * function behaves the same way as the SYCL constructor.
 *
 * @param    DRef           An opaque pointer to a ``sycl::device``.
 * @param    handler        A callback function that will be invoked by the
 *                          async_handler used during queue creation. Can be
 *                          NULL if no async_handler is needed.
 * @param    properties     A combination of bit flags using the values defined
 *                          in the DPCTLQueuePropertyType enum. The bit flags
 *                          are used to create a ``sycl::property_list`` that is
 *                          passed to the SYCL queue constructor.
 * @return An opaque DPCTLSyclQueueRef pointer containing the new
 * ``sycl::queue`` object. A nullptr is returned if the queue could not be
 * created.
 * @ingroup QueueInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclQueueRef
DPCTLQueue_CreateForDevice(__dpctl_keep const DPCTLSyclDeviceRef DRef,
                           error_handler_callback *handler,
                           int properties);

/*!
 * @brief Delete the pointer after casting it to sycl::queue.
 *
 * @param    QRef           A DPCTLSyclQueueRef pointer that gets deleted.
 * @ingroup QueueInterface
 */
DPCTL_API
void DPCTLQueue_Delete(__dpctl_take DPCTLSyclQueueRef QRef);

/*!
 * @brief Returns a copy of the DPCTLSyclQueueRef object.
 *
 * @param    QRef           DPCTLSyclQueueRef object to be copied.
 * @return   A new DPCTLSyclQueueRef created by copying the passed in
 * DPCTLSyclQueueRef object.
 * @ingroup QueueInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclQueueRef
DPCTLQueue_Copy(__dpctl_keep const DPCTLSyclQueueRef QRef);

/*!
 * @brief Checks if two DPCTLSyclQueueRef objects point to the
 * same ``sycl::queue``.
 *
 * @param    QRef1          First opaque pointer to the ``sycl::queue``.
 * @param    QRef2          Second opaque pointer to the ``sycl::queue``.
 * @return   True if the underlying sycl::queue are same, false otherwise.
 * @ingroup QueueInterface
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
 * @ingroup QueueInterface
 */
DPCTL_API
DPCTLSyclBackendType DPCTLQueue_GetBackend(__dpctl_keep DPCTLSyclQueueRef QRef);

/*!
 * @brief Returns the Sycl context for the queue.
 *
 * @param    QRef           An opaque pointer to the sycl queue.
 * @return   A DPCTLSyclContextRef pointer to the sycl context for the queue.
 * @ingroup QueueInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclContextRef
DPCTLQueue_GetContext(__dpctl_keep const DPCTLSyclQueueRef QRef);

/*!
 * @brief returns the Sycl device for the queue.
 *
 * @param    QRef           An opaque pointer to the sycl queue.
 * @return   A DPCTLSyclDeviceRef pointer to the sycl device for the queue.
 * @ingroup QueueInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclDeviceRef
DPCTLQueue_GetDevice(__dpctl_keep const DPCTLSyclQueueRef QRef);

/*! @brief Structure to be used to specify dimensionality and type of
 * local_accessor kernel type argument.
 */
typedef struct MDLocalAccessorTy
{
    size_t ndim;
    DPCTLKernelArgType dpctl_type_id;
    size_t dim0;
    size_t dim1;
    size_t dim2;
} MDLocalAccessor;

/*!
 * @brief Submits the kernel to the specified queue with the provided range
 * argument.
 *
 * A wrapper over ``sycl::queue.submit()``. The function takes an
 * interoperability kernel, the kernel arguments, and a ``sycl::queue`` as
 * input. The kernel is submitted as
 * ``parallel_for(range<NRange>, *unwrap<kernel>(KRef))``.
 *
 * \todo ``sycl::buffer`` arguments are not supported yet.
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
 *                          for the kernel. We call ``sycl::handler.depends_on``
 *                          for each of the provided events.
 * @param    NDepEvents     Size of the DepEvents list.
 * @return   An opaque pointer to the ``sycl::event`` returned by the
 *           ``sycl::queue.submit()`` function.
 * @ingroup QueueInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclEventRef
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
 * A wrapper over ``sycl::queue.submit()``. The function takes an
 * interoperability kernel, the kernel arguments, and a Sycl queue as input.
 * The kernel is submitted as
 * ``parallel_for(nd_range<NRange>, *unwrap<kernel>(KRef))``.
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
 *                          for the kernel. We call ``sycl::handler.depends_on``
 *                          for each of the provided events.
 * @param    NDepEvents     Size of the DepEvents list.
 * @return   An opaque pointer to the ``sycl::event`` returned by the
 *           ``sycl::queue.submit()`` function.
 * @ingroup QueueInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclEventRef
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
 * @brief Calls the ``sycl::queue::submit`` function to do a blocking wait on
 * all enqueued tasks in the queue.
 *
 * @param    QRef           Opaque pointer to a ``sycl::queue``.
 * @ingroup QueueInterface
 */
DPCTL_API
void DPCTLQueue_Wait(__dpctl_keep const DPCTLSyclQueueRef QRef);

/*!
 * @brief C-API wrapper for ``sycl::queue::memcpy``.
 *
 * @param    QRef           An opaque pointer to the ``sycl::queue``.
 * @param    Dest           An USM pointer to the destination memory.
 * @param    Src            An USM pointer to the source memory.
 * @param    Count          A number of bytes to copy.
 * @return   An opaque pointer to the ``sycl::event`` returned by the
 *           ``sycl::queue::memcpy`` function.
 * @ingroup QueueInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclEventRef
DPCTLQueue_Memcpy(__dpctl_keep const DPCTLSyclQueueRef QRef,
                  void *Dest,
                  const void *Src,
                  size_t Count);

/*!
 * @brief C-API wrapper for ``sycl::queue::memcpy``.
 *
 * @param    QRef           An opaque pointer to the ``sycl::queue``.
 * @param    Dest           An USM pointer to the destination memory.
 * @param    Src            An USM pointer to the source memory.
 * @param    Count          A number of bytes to copy.
 * @param    DepEvents      A pointer to array of DPCTLSyclEventRef opaque
 *                          pointers to dependent events.
 * @param    DepEventsCount A number of dependent events.
 * @return   An opaque pointer to the ``sycl::event`` returned by the
 *           ``sycl::queue::memcpy`` function.
 * @ingroup QueueInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclEventRef
DPCTLQueue_MemcpyWithEvents(__dpctl_keep const DPCTLSyclQueueRef QRef,
                            void *Dest,
                            const void *Src,
                            size_t Count,
                            __dpctl_keep const DPCTLSyclEventRef *DepEvents,
                            size_t DepEventsCount);

/*!
 * @brief C-API wrapper for ``sycl::queue::prefetch``.
 *
 * @param    QRef           An opaque pointer to the ``sycl::queue``.
 * @param    Ptr            An USM pointer to memory.
 * @param    Count          A number of bytes to prefetch.
 * @return   An opaque pointer to the ``sycl::event`` returned by the
 *           ``sycl::queue::prefetch`` function.
 * @ingroup QueueInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclEventRef
DPCTLQueue_Prefetch(__dpctl_keep DPCTLSyclQueueRef QRef,
                    const void *Ptr,
                    size_t Count);

/*!
 * @brief C-API wrapper for ``sycl::queue::mem_advise``.
 *
 * @param    QRef           An opaque pointer to the ``sycl::queue``.
 * @param    Ptr            An USM pointer to memory.
 * @param    Count          A number of bytes to prefetch.
 * @param    Advice         Device-defined advice for the specified allocation.
 *                          A value of 0 reverts the advice for Ptr to the
 *                          default behavior.
 * @return   An opaque pointer to the ``sycl::event`` returned by the
 *           ``sycl::queue::mem_advise`` function.
 * @ingroup QueueInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclEventRef
DPCTLQueue_MemAdvise(__dpctl_keep DPCTLSyclQueueRef QRef,
                     const void *Ptr,
                     size_t Count,
                     int Advice);

/*!
 * @brief C-API wrapper for sycl::queue::is_in_order that indicates whether
 * the referenced queue is in-order or out-of-order.
 *
 * @param    QRef         An opaque pointer to the ``sycl::queue``.
 * @ingroup QueueInterface
 */
DPCTL_API
bool DPCTLQueue_IsInOrder(__dpctl_keep const DPCTLSyclQueueRef QRef);

/*!
 * @brief C-API wrapper for
 * sycl::queue::has_property<sycl::property::queue::enable_profiling>() that
 * indicates whether the referenced queue was constructed with this property.
 *
 * @param    QRef         An opaque pointer to the ``sycl::queue``.
 * @ingroup QueueInterface
 */
DPCTL_API
bool DPCTLQueue_HasEnableProfiling(__dpctl_keep const DPCTLSyclQueueRef QRef);

/*!
 * @brief C-API wrapper for std::hash<sycl::queue>'s operator().
 *
 * @param    QRef         An opaque pointer to the ``sycl::queue``.
 * @return   Hash value of the underlying ``sycl::queue`` instance.
 * @ingroup QueueInterface
 */
DPCTL_API
size_t DPCTLQueue_Hash(__dpctl_keep const DPCTLSyclQueueRef QRef);

/*!
 * @brief C-API wrapper for ``sycl::queue::submit_barrier()``.
 *
 * @param    QRef    An opaque pointer to the ``sycl::queue``.
 * @return   An opaque pointer to the ``sycl::event`` returned by the
 *           ``sycl::queue::submit_barrier()`` function.
 */
DPCTL_API
__dpctl_give DPCTLSyclEventRef
DPCTLQueue_SubmitBarrier(__dpctl_keep const DPCTLSyclQueueRef QRef);

/*!
 * @brief C-API wrapper for ``sycl::queue::submit_barrier(event_vector)``.
 *
 * @param    QRef    An opaque pointer to the ``sycl::queue``.
 * @param    DepEvents     List of dependent DPCTLSyclEventRef objects (events)
 *                         for the barrier. We call ``sycl::handler.depends_on``
 *                         for each of the provided events.
 * @param    NDepEvents    Size of the DepEvents list.
 * @return   An opaque pointer to the ``sycl::event`` returned by the
 *           ``sycl::queue::submit_barrier()`` function.
 */
DPCTL_API
__dpctl_give DPCTLSyclEventRef DPCTLQueue_SubmitBarrierForEvents(
    __dpctl_keep const DPCTLSyclQueueRef QRef,
    __dpctl_keep const DPCTLSyclEventRef *DepEvents,
    size_t NDepEvents);

/*!
 * @brief C-API wrapper for ``sycl::queue::memset``.
 *
 * @param    QRef           An opaque pointer to the ``sycl::queue``.
 * @param    USMRef         An USM pointer to the memory to fill.
 * @param    Value          A value to fill.
 * @param    Count          A number of uint8_t elements to fill.
 * @return   An opaque pointer to the ``sycl::event`` returned by the
 *           ``sycl::queue::fill`` function.
 * @ingroup QueueInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclEventRef
DPCTLQueue_Memset(__dpctl_keep const DPCTLSyclQueueRef QRef,
                  void *USMRef,
                  uint8_t Value,
                  size_t Count);

/*!
 * @brief C-API wrapper for ``sycl::queue::fill``.
 *
 * @param    QRef           An opaque pointer to the ``sycl::queue``.
 * @param    USMRef         An USM pointer to the memory to fill.
 * @param    Value          A uint8_t value to fill.
 * @param    Count          A number of uint8_t elements to fill.
 * @return   An opaque pointer to the ``sycl::event`` returned by the
 *           ``sycl::queue::fill`` function.
 * @ingroup QueueInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclEventRef
DPCTLQueue_Fill8(__dpctl_keep const DPCTLSyclQueueRef QRef,
                 void *USMRef,
                 uint8_t Value,
                 size_t Count);

/*!
 * @brief C-API wrapper for ``sycl::queue::fill``.
 *
 * @param    QRef           An opaque pointer to the ``sycl::queue``.
 * @param    USMRef         An USM pointer to the memory to fill.
 * @param    Value          A uint16_t value to fill.
 * @param    Count          A number of uint16_t elements to fill.
 * @return   An opaque pointer to the ``sycl::event`` returned by the
 *           ``sycl::queue::fill`` function.
 * @ingroup QueueInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclEventRef
DPCTLQueue_Fill16(__dpctl_keep const DPCTLSyclQueueRef QRef,
                  void *USMRef,
                  uint16_t Value,
                  size_t Count);

/*!
 * @brief C-API wrapper for ``sycl::queue::fill``.
 *
 * @param    QRef           An opaque pointer to the ``sycl::queue``.
 * @param    USMRef         An USM pointer to the memory to fill.
 * @param    Value          A uint32_t value to fill.
 * @param    Count          A number of uint32_t elements to fill.
 * @return   An opaque pointer to the ``sycl::event`` returned by the
 *           ``sycl::queue::fill`` function.
 * @ingroup QueueInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclEventRef
DPCTLQueue_Fill32(__dpctl_keep const DPCTLSyclQueueRef QRef,
                  void *USMRef,
                  uint32_t Value,
                  size_t Count);

/*!
 * @brief C-API wrapper for ``sycl::queue::fill``.
 *
 * @param    QRef           An opaque pointer to the ``sycl::queue``.
 * @param    USMRef         An USM pointer to the memory to fill.
 * @param    Value          A uint64_t value to fill.
 * @param    Count          A number of uint64_t elements to fill.
 * @return   An opaque pointer to the ``sycl::event`` returned by the
 *           ``sycl::queue::fill`` function.
 * @ingroup QueueInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclEventRef
DPCTLQueue_Fill64(__dpctl_keep const DPCTLSyclQueueRef QRef,
                  void *USMRef,
                  uint64_t Value,
                  size_t Count);

/*!
 * @brief C-API wrapper for ``sycl::queue::fill``.
 *
 * @param    QRef           An opaque pointer to the ``sycl::queue``.
 * @param    USMRef         An USM pointer to the memory to fill.
 * @param    Value          A pointer to uint64_t array of 2 elements with value
 * to fill.
 * @param    Count          A number of 128-bit elements to fill.
 * @return   An opaque pointer to the ``sycl::event`` returned by the
 *           ``sycl::queue::fill`` function.
 * @ingroup QueueInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclEventRef
DPCTLQueue_Fill128(__dpctl_keep const DPCTLSyclQueueRef QRef,
                   void *USMRef,
                   uint64_t *Value,
                   size_t Count);

DPCTL_C_EXTERN_C_END
