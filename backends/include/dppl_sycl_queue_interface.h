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
/// This header declares a C interface to DPPL's sycl::queue manager class that
/// maintains a thread local stack of sycl::queues objects for use inside
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
#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"

DPPL_C_EXTERN_C_BEGIN

/*!
 * @brief Redefinition of Sycl's device_type so that we do not have to include
 * sycl.hpp here and in the Python bindings.
 *
 */
typedef enum
{
    DPPL_CPU,
    DPPL_GPU,
    DPPL_ACCELERATOR,
    DPPL_CUSTOM,
    DPPL_AUTOMATIC,
    DPPL_HOST,
    DPPL_ALL
} DPPLSyclDeviceType;


////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// DpplSyclQueueManager ///////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
 * @brief Get the number of sycl::platform available on the system.
 *
 * @return The number of available sycl::platforms.
 */
DPPL_API
size_t DPPLGetNumPlatforms ();

/*!
 * @brief Get the sycl::queue object that is currently activated for this
 * thread.
 *
 * @return A copy of the current (top of the stack) sycl::queue is returned
 * wrapped inside an opaque DPPLSyclQueueRef pointer.
 */
DPPL_API
__dppl_give DPPLSyclQueueRef DPPLGetCurrentQueue ();

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
__dppl_give DPPLSyclQueueRef DPPLGetQueue (DPPLSyclDeviceType DeviceTy,
                                           size_t DNum);

/*!
 * @brief Get the number of activated queues not including the global or
 * default queue.
 *
 * @return The number of activated queues.
 */
DPPL_API
size_t DPPLGetNumActivatedQueues ();

/*!
 * @brief Get the number of GPU queues available on the system.
 *
 * @return The number of available GPU queues.
 */
DPPL_API
size_t DPPLGetNumGPUQueues ();

/*!
 * @brief Get the number of CPU queues available on the system.
 *
 * @return The number of available CPU queues.
 */
DPPL_API
size_t DPPLGetNumCPUQueues ();

/*!
* @brief Set the default DPPL queue to the sycl::queue for the given device.
*
* If no such device is found the a runtime_error exception is thrown.
*
* @param    DeviceTy       The type of Sycl device (sycl_device_type)
* @param    DNum           Device id for the device (defaults to 0)
*/
DPPL_API
void DPPLSetAsDefaultQueue (DPPLSyclDeviceType DeviceTy,
                            size_t DNum);

/*!
 * @brief Sets as the sycl queue corresponding to the specified device as
 * the currently active DPPL queue, and returns a copy to the queue to
 * the caller.
 *
 * @param    DeviceTy       The type of Sycl device (sycl_device_type)
 * @param    DNum           Device id for the device (defaults to 0)
 *
 * @return A copy of the sycl::queue corresponding to the current queue for
 * the thread is returned wrapped inside a DPPLSyclDeviceType pointer. A
 * runtime_error exception is thrown if no current queue was found (can only
 * happen is somehow the stack got corrupted, since a default queue should
 * always exist).
 */
DPPL_API
__dppl_give DPPLSyclQueueRef DPPLSetAsCurrentQueue (DPPLSyclDeviceType DeviceTy,
                                                    size_t DNum);

/*!
 * @brief The current DPPL queue is popped from the stack of activated
 * queues, except in the scenario where the current queue is the default
 * queue.
 */
DPPL_API
void DPPLRemoveCurrentQueue ();

/*!
 * @brief Prints out information about the Sycl environment, such as
 * number of available platforms, number of activated queues, etc.
 */
DPPL_API
void DPPLDumpPlatformInfo ();

/*!
 * @brief Prints out information about the device corresponding to the
 * sycl::queue argument.
 *
 * @param    QRef           A DPPLSyclQueueRef pointer whose metadata will be
 *                          printed out.
 */
DPPL_API
void DPPLDumpDeviceInfo (__dppl_keep const DPPLSyclQueueRef QRef);

/*!
 * @brief Delete the pointer after static casting it to sycl::queue.
 *
 * @param    QRef           A DPPLSyclQueueRef pointer that gets deleted.
 */
DPPL_API
void DPPLDeleteQueue (__dppl_take DPPLSyclQueueRef QRef);

DPPL_C_EXTERN_C_END
