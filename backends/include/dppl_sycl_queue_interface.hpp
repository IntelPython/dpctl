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

#include <cstdlib>
#include <deque>

#ifdef _WIN32
#    ifdef DPPLSyclInterface_EXPORTS
#        define DPPL_API __declspec(dllexport)
#    else
#        define DPPL_API __declspec(dllimport)
#    endif
#else
#    define DPPL_API
#endif

namespace dppl
{

/*!
 * @brief Redefinition of Sycl's device_type so that we do not have to include
 * sycl.hpp here and in the Python bindings.
 *
 */
enum class sycl_device_type : unsigned int
{
    cpu,
    gpu,
    accelerator,
    custom,
    automatic,
    host,
    all
};


////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// DpplSyclQueueManager ///////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
 * @brief The class exposes factory methods to get handles to sycl::queues,
 * mainatins a thread local stack of sycl::queue objects, and other
 * functionalitites to manager sycl::queue objects.
 *
 */
class DpplSyclQueueManager
{
public:
    /*!
     * @brief Get the number of sycl::platform avilable on the system.
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
    * @brief Set the global DPPL queue to the sycl::queue for the given device.
    *
    *
    * @param    DeviceTy       The type of Sycl device (sycl_device_type)
    * @param    DNum           Device id for the device (defaults to 0)
    * @return   error_code     DPPL_SUCCESS if the global queue was successfully
    *                          set to the asked for device, otherwise
    *                          DPPL_FAILURE.
    */
    DPPL_API
    int64_t setAsGlobalQueue (dppl::sycl_device_type DeviceTy,
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
     * @brief The current DPPL queue is deactivated and any previously activated
     * queue, including the global queue, becomes the new active DPPL queue.
     *
     * @return   error_code     DPPL_SUCCESS if the current queue was
     *                          successfully removed, otherwise DPPL_FAILURE.
     */
    DPPL_API
    int64_t removeCurrentQueue ();

    /*!
     * @brief
     *
     * @return   {return}       My Param doc
     */
    DPPL_API
    int64_t dump () const;

    /*!
     * @brief
     *
     * @param    QPtr           My Param doc
     * @return   {return}       My Param doc
     */
    DPPL_API
    int64_t dump_queue (const void *QPtr) const;
};

/*!
 * @brief
 *
 * @param    QPtr           My Param doc
 * @return   {return}       My Param doc
 */
DPPL_API
int64_t deleteQueue (void *QPtr);

} /* end of namespace dppl */
