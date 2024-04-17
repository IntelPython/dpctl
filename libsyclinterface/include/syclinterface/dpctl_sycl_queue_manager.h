//===---- dpctl_sycl_queue_manager.h - A manager for sycl queues -*-C++-*- ===//
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
/// This header declares a set of functions to support a concept of current
/// queue for applications using dpctl.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"
#include "dpctl_data_types.h"
#include "dpctl_sycl_types.h"

DPCTL_C_EXTERN_C_BEGIN

/**
 * @defgroup QueueManager Queue class helper functions
 */

/*!
 * @brief Get the current sycl::queue for the thread of execution.
 *
 * Dpctl lets an application access a "current queue" as soon as the application
 * loads dpctl. The initial current queue also termed the global queue is a
 * queue created using SYCL's default_selector. The current queue is set per
 * thread and can be changed for a specific execution scope using the PushQueue
 * and PopQueue functions in this module. The global queue can also be changed
 * by using SetGlobalQueue.
 *
 * The DPCTLQueueMgr_GetCurrentQueue function returns the current queue in the
 * current scope from where the function was called.
 *
 * @return An opaque DPCTLSyclQueueRef pointer wrapping a sycl::queue*.
 * @ingroup QueueManager
 */
DPCTL_API
__dpctl_give DPCTLSyclQueueRef DPCTLQueueMgr_GetCurrentQueue(void);

/*!
 * @brief Returns true if the global queue set for the queue manager is also the
 * current queue.
 *
 * The default current queue provided by the queue manager is termed as the
 * global queue. If DPCTLQueueMgr_PushQueue is used to make another queue the
 * current queue, then the global queue no longer remains the current queue till
 * all pushed queues are popped using DPCTLQueueMgr_PopQueue. The
 * DPCTLQueueMgr_GlobalQueueIsCurrent checks if the global queue is also the
 * current queue, i.e., no queues have been pushed and are yet to be popped.
 *
 * @return True if the global queue is the current queue, else false.
 * @ingroup QueueManager
 */
DPCTL_API
bool DPCTLQueueMgr_GlobalQueueIsCurrent(void);

/*!
 * @brief Check if the queue argument is also the current queue.
 *
 * @param    QRef           An opaque pointer to a sycl::queue.
 * @return   True if QRef argument is the the current queue, else False.
 * @ingroup QueueManager
 */
DPCTL_API
bool DPCTLQueueMgr_IsCurrentQueue(__dpctl_keep const DPCTLSyclQueueRef QRef);

/*!
 * @brief Resets the global queue using the passed in DPCTLSyclQueueRef the
 * previous global queue is deleted.
 *
 * @param    QRef           An opaque reference to a sycl::device.
 * @ingroup QueueManager
 */
DPCTL_API
void DPCTLQueueMgr_SetGlobalQueue(__dpctl_keep const DPCTLSyclQueueRef QRef);

/*!
 * @brief Pushes the passed in ``sycl::queue`` object to the queue manager's
 * internal stack of queues and makes the queue the current queue.
 *
 * The queue manager maintains a thread-local stack of sycl::queue
 * objects. The DPCTLQueueMgr_PushQueue() function pushes to the stack and sets
 * the passed in DPCTLSyclQueueRef object as the current queue. The
 * current queue is the queue returned by the DPCTLQueueMgr_GetCurrentQueue()
 * function.
 *
 * @param    QRef           An opaque reference to a ``sycl::queue``.
 * @ingroup QueueManager
 */
DPCTL_API
void DPCTLQueueMgr_PushQueue(__dpctl_keep const DPCTLSyclQueueRef QRef);

/*!
 * @brief Pops the top of stack sycl::queue object from the queue manager's   *
 * internal stack of queues and makes the next queue in the stack the current
 * queue.
 *
 * DPCTLPopSyclQueue removes the top of stack queue and changes the
 * current queue. If no queue was previously pushed, then a
 * DPCTLQueueMgr_PopQueue call is a no-op.
 * @ingroup QueueManager
 */
DPCTL_API
void DPCTLQueueMgr_PopQueue(void);

/*!
 * @brief A helper function meant for unit testing. Returns the current number
 * of queues pushed to the queue manager's internal stack of sycl::queue
 * objects.
 *
 * @return   The current size of the queue manager's stack of queues.
 * @ingroup QueueManager
 */
DPCTL_API
size_t DPCTLQueueMgr_GetQueueStackSize(void);

DPCTL_C_EXTERN_C_END
