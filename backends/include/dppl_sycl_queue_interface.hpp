//===--- dppl_sycl_queue_interface.hpp - DPPL-SYCL interface ---*- C++ -*---===//
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
/// This file contains the wrapper functions to manage SYCL queue objects.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "dppl_macros.hpp"
#include <cstdlib>

namespace dppl
{

/*!
 * @brief Redefinition of Sycl's device_type so that we do not have to include
 * sycl.hpp here and in the Python bindings.
 *
 */
enum class sycl_device_type : unsigned int
{
    DPPL_CPU,
    DPPL_GPU,
    DPPL_ACCELERATOR,
    DPPL_CUSTOM,
    DPPL_AUTOMATIC,
    DPPL_HOST,
    DPPL_ALL
};


////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// DpplSyclQueueManager ///////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
 * @brief The class exposes factory methods to get handles to sycl::queues,
 * mainatins a thread local stack of sycl::queue objects, and other
 * functionalities to manager sycl::queue objects.
 *
 */
class DpplSyclQueueManager
{
public:
    /*!
     * @brief Get the number of sycl::platform available on the system.
     *
     * @param    platforms      The number of available sycl::platforms is
     *                          stored into this param.
     * @return   error_code     DPPL_SUCCESS if a platforms was updated
     *                          otherwise DPPL_FAILURE.
     */
    DPPL_API
    int64_t getNumPlatforms (size_t &platforms) const;

    /*!
     * @brief Get the sycl::queue object that is currently activated for this
     * thread.
     *
     * @param    Ptr2QPtr       The *Ptr2QPtr is allocated with a copy of the
     *                          current queue.
     * @return   error_code     DPPL_SUCCESS if a queue exists or DPPL_FAILURE
     *                          if no queue was currently activated.
     */
    DPPL_API
    int64_t getCurrentQueue (void **Ptr2QPtr) const;

    /*!
     * @brief Get a sycl::queue object of the specified type and device id.
     *
     * @param    Ptr2QPtr       The *Ptr2QPtr is allocated with a copy of the
     *                          sycl queue corresponding to this device.
     * @param    DeviceTy       The type of Sycl device (sycl_device_type)
     * @param    DNum           Device id for the device (defaults to 0)
     * @return   error_code     DPPL_SUCCESS if a queue exists or DPPL_FAILURE
     *                          if no queue of asked for type and id was found.
     */
    DPPL_API
    int64_t getQueue (void **Ptr2QPtr,
                      dppl::sycl_device_type DeviceTy,
                      size_t DNum = 0) const;

    /*!
     * @brief Get the number of activated queues not including the global or
     * default queue.
     *
     * @param    numQueues      Populated with the number of activated queues
     *                          and returned to caller.
     * @return   error_code     DPPL_SUCCESS or DPPL_FAILURE
     */
    DPPL_API
    int64_t getNumActivatedQueues (size_t &numQueues) const;

    /*!
     * @brief Get the number of GPU queues available on the system.
     *
     * @param    numQueues      Populated with the number of available GPU
     *                          queues and returned to caller.
     * @return   error_code     DPPL_SUCCESS or DPPL_FAILURE
     */
    DPPL_API
    int64_t getNumGPUQueues (size_t &numQueues) const;

    /*!
     * @brief Get the number of CPU queues available on the system.
     *
     * @param    numQueues      Populated with the number of available CPU
     *                          queues and returned to caller.
     * @return   error_code     DPPL_SUCCESS or DPPL_FAILURE
     */
    DPPL_API
    int64_t getNumCPUQueues (size_t &numQueues) const;

    /*!
    * @brief Set the default DPPL queue to the sycl::queue for the given device.
    *
    *
    * @param    DeviceTy       The type of Sycl device (sycl_device_type)
    * @param    DNum           Device id for the device (defaults to 0)
    * @return   error_code     DPPL_SUCCESS if the default queue was
    *                          successfully set to the asked for device,
    *                          otherwise DPPL_FAILURE.
    */
    DPPL_API
    int64_t setAsDefaultQueue (dppl::sycl_device_type DeviceTy,
                               size_t DNum = 0);

    /*!
     * @brief Sets as the sycl queue corresponding to the specified device as
     * the currently active DPPL queue, and returns a handle to the queue to
     * the caller.
     *
     * @param    Ptr2QPtr       The *Ptr2QPtr is allocated with a copy of the
     *                          sycl queue corresponding to this device.
     * @param    DeviceTy       The type of Sycl device (sycl_device_type)
     * @param    DNum           Device id for the device (defaults to 0)
     * @return   error_code     DPPL_SUCCESS if the queue was successfully
     *                          created, otherwise DPPL_FAILURE.
     */
    DPPL_API
    int64_t setAsCurrentQueue (void **Ptr2QPtr,
                               dppl::sycl_device_type DeviceTy,
                               size_t DNum);

    /*!
     * @brief The current DPPL queue is popped from the stack of activated
     * queues, except in the scenario where the current queue is the default
     * queue.
     *
     * @return   error_code     DPPL_SUCCESS if the current queue was
     *                          successfully removed, otherwise DPPL_FAILURE.
     */
    DPPL_API
    int64_t removeCurrentQueue ();

    /*!
     * @brief Prints out information about the Sycl environment, such as
     * number of available platforms, number of activated queues, etc.
     *
     * @return   error_code     DPPL_SUCCESS if the metadata for the queue
     *                          manager was successfully printed out, otherwise
     *                          DPPL_FAILURE.
     */
    DPPL_API
    int64_t dump () const;

    /*!
     * @brief Prints out information about the device corresponding to the
     * sycl::queue argument.
     *
     * @param    QPtr           Pointer to a sycl::queue.
     * @return   error_code     DPPL_SUCCESS if the metadata for the queue
     *                          was successfully printed out, otherwise
     *                          DPPL_FAILURE.
     */
    DPPL_API
    int64_t dumpDeviceInfo (const void *QPtr) const;
};

/*!
 * @brief Delete the pointer after static casting it to sycl::queue.
 *
 * @param    QPtr           Pointer to a sycl::queue.
 * @return   error_code     DPPL_SUCCESS if the pointer was deleted, otherwise
 *                          DPPL_FAILURE.
 */
DPPL_API
int64_t deleteQueue (void *QPtr);

} /* end of namespace dppl */
