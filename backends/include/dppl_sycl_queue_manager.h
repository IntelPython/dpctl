//===----------- dppl_sycl_queue_manager.h - dpctl-C_API ---*---C++ ---*---===//
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
/// This header declares a C interface to DPPL's sycl::queue manager to
/// maintain a thread local stack of sycl::queues objects for use inside
/// Python programs. The C interface is designed in a way to not have to
/// include the Sycl headers inside a Python extension module, since that would
/// require the extension to be compiled using dpc++ or another Sycl compiler.
/// Compiling the extension with a compiler different from what was used to
/// compile the Python interpreter can cause run-time problems especially on MS
/// Windows. Additionally, the C interface makes it easier to interoperate with
/// Numba without having to deal with C++ name mangling.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "dppl_data_types.h"
#include "dppl_sycl_types.h"
#include "dppl_sycl_context_interface.h"
#include "dppl_sycl_device_interface.h"
#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"

DPPL_C_EXTERN_C_BEGIN

/*!
 * @brief Get the sycl::queue object that is currently activated for this
 * thread.
 *
 * @return A copy of the current (top of the stack) sycl::queue is returned
 * wrapped inside an opaque DPPLSyclQueueRef pointer.
 */
DPPL_API
__dppl_give DPPLSyclQueueRef DPPLQueueMgr_GetCurrentQueue ();

/*!
 * @brief Get a sycl::queue object of the specified type and device id.
 *
 * @param    DeviceTy       The type of Sycl device (sycl_device_type)
 * @param    DNum           Device id for the device (defaults to 0)
 *
 * @return A copy of the sycl::queue corresponding to the device is returned
 * wrapped inside a DPPLSyclDeviceType pointer. A runtime_error exception is
 * raised if no such device exists.
 */
DPPL_API
__dppl_give DPPLSyclQueueRef
DPPLQueueMgr_GetQueue (DPPLSyclBackendType BETy,
                       DPPLSyclDeviceType DeviceTy,
                       size_t DNum);

/*!
 * @brief Get the number of activated queues not including the global or
 * default queue.
 *
 * @return The number of activated queues.
 */
DPPL_API
size_t DPPLQueueMgr_GetNumActivatedQueues ();


/*!
 * @brief Get the number of available queues for given backend and device type
 * combination.
 *
 * @param    BETy           Type of Sycl backend.
 * @param    DeviceTy       Type of Sycl device.
 * @return   The number of available queues.
 */
DPPL_API
size_t DPPLQueueMgr_GetNumQueues (DPPLSyclBackendType BETy,
                                  DPPLSyclDeviceType DeviceTy);

/*!
 * @brief Returns True if the passed in queue and the current queue are the
 * same, else returns False.
 *
 * @param    QRef           An opaque pointer to a sycl::queue.
 * @return   True or False depending on whether the QRef argument is the same as
 * the currently activated queue.
 */
DPPL_API
bool DPPLQueueMgr_IsCurrentQueue (__dppl_keep const DPPLSyclQueueRef QRef);

/*!
* @brief Set the default DPPL queue to the sycl::queue for the given backend
* and device type combination and return a DPPLSyclQueueRef for that queue.
* If no queue was created Null is returned to caller.
*
* @param    BETy           Type of Sycl backend.
* @param    DeviceTy       The type of Sycl device (sycl_device_type)
* @param    DNum           Device id for the device
* @return A copy of the sycl::queue that was set as the new default queue. If no
* queue could be created then returns Null.
*/
DPPL_API
__dppl_give DPPLSyclQueueRef
DPPLQueueMgr_SetAsDefaultQueue (DPPLSyclBackendType BETy,
                                DPPLSyclDeviceType DeviceTy,
                                size_t DNum);

/*!
 * @brief Pushes a new sycl::queue object to the top of DPPL's thread-local
 * stack of a "activated" queues, and returns a copy of the queue to caller.
 *
 * The DPPL queue manager maintains a thread-local stack of sycl::queue objects
 * to facilitate nested parallelism. The sycl::queue at the top of the stack is
 * termed as the currently activated queue, and is always the one returned by
 * DPPLQueueMgr_GetCurrentQueue(). DPPLPushSyclQueueToStack creates a new
 * sycl::queue corresponding to the specified device and pushes it to the top
 * of the stack. A copy of the sycl::queue is returned to the caller wrapped
 * inside the opaque DPPLSyclQueueRef pointer. A runtime_error exception is
 * thrown when a new sycl::queue could not be created for the specified device.
 *
 * @param    BETy           Type of Sycl backend.
 * @param    DeviceTy       The type of Sycl device (sycl_device_type)
 * @param    DNum           Device id for the device (defaults to 0)
 *
 * @return A copy of the sycl::queue that was pushed to the top of DPPL's
 * stack of sycl::queue objects. Nullptr is returned if no such device exists.
 */
DPPL_API
__dppl_give DPPLSyclQueueRef
DPPLQueueMgr_PushQueue (DPPLSyclBackendType BETy,
                        DPPLSyclDeviceType DeviceTy,
                        size_t DNum);

/*!
 * @brief Pops the top of stack element from DPPL's stack of activated
 * sycl::queue objects.
 *
 * DPPLPopSyclQueue only removes the reference from the DPPL stack of
 * sycl::queue objects. Any instance of the popped queue that were previously
 * acquired by calling DPPLPushSyclQueue() or DPPLQueueMgr_GetCurrentQueue()
 * needs to be freed separately. In addition, a runtime_error is thrown when
 * the stack contains only one sycl::queue, i.e., the default queue.
 *
 */
DPPL_API
void DPPLQueueMgr_PopQueue ();

DPPL_C_EXTERN_C_END
