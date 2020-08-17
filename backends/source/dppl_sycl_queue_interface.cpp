//===--- dppl_sycl_queue_interface.cpp - DPPL-SYCL interface --*- C++ -*---===//
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
/// This file implements the data types and functions declared in
/// dppl_sycl_queue_interface.hpp.
///
//===----------------------------------------------------------------------===//
#include "dppl_sycl_queue_interface.hpp"
#include "dppl_error_codes.hpp"
#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <CL/sycl.hpp>                /* SYCL headers   */

using namespace cl::sycl;
using namespace dppl;

/*------------------------------- Private helpers ----------------------------*/

// Anonymous namespace for private helpers
namespace
{
/*!
 * @brief
 *
 * @param    Device         My Param doc
 */
void dump_device_info (const device & Device)
{
    std::stringstream ss;

    ss << std::setw(4) << " " << std::left << std::setw(16) << "Name"
       << Device.get_info<info::device::name>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(16) << "Driver version"
       << Device.get_info<info::device::driver_version>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(16) << "Vendor"
       << Device.get_info<info::device::vendor>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(16) << "Profile"
       << Device.get_info<info::device::profile>() << '\n';

    std::cout << ss.str();
}

/*!
 * @brief
 *
 * @param    Platform       My Param doc
 */
void dump_platform_info (const platform & Platform)
{
    std::stringstream ss;

    ss << std::setw(4) << " " << std::left << std::setw(12) << "Name"
       << Platform.get_info<info::platform::name>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(12) << "Version"
       << Platform.get_info<info::platform::version>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(12) << "Vendor"
       << Platform.get_info<info::platform::vendor>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(12) << "Profile"
       << Platform.get_info<info::platform::profile>() << '\n';

    std::cout << ss.str();
}

int64_t error_reporter (const std::string & msg)
{
    std::cerr << "Error: " << msg << '\n';
    return DPPL_FAILURE;
}

/*!
 * @brief
 *
 */
class QMgrHelper
{
public:
    static std::vector<cl::sycl::queue>   cpu_queues;
    static std::vector<cl::sycl::queue>   gpu_queues;
    static thread_local std::vector<cl::sycl::queue> active_queues;
    /*!
     * @brief Get the Queue object
     *
     * @param    QPtr           My Param doc
     * @param    DeviceTy       My Param doc
     * @param    DNum           My Param doc
     * @return   {return}       My Param doc
     */
    static int64_t
    getQueue (void **QPtr, sycl_device_type DeviceTy, size_t DNum);

    /*!
     * @brief Get the Current Queue object
     *
     * @param    q              My Param doc
     * @return   {return}       My Param doc
     */
    static int64_t
    getCurrentQueue (void **q);

    /*!
    * @brief Set the As Global Queue object
    *
    * @param    DeviceTy       My Param doc
    * @param    DNum           My Param doc
    * @return   {return}       My Param doc
    */
    static int64_t
    setAsGlobalQueue (sycl_device_type DeviceTy, size_t DNum);

    /*!
     * @brief Set the As Current Queue object
     *
     * @param    QPtr           My Param doc
     * @param    DeviceTy       My Param doc
     * @param    DNum           My Param doc
     * @return   {return}       My Param doc
     */
    static int64_t
    setAsCurrentQueue (void **QPtr,
                       dppl::sycl_device_type DeviceTy,
                       size_t DNum);

    /*!
     * @brief
     *
     * @return   {return}       My Param doc
     */
    static int64_t
    removeCurrentQueue ();

    /*!
     * @brief
     *
     * @param    device_ty      My Param doc
     * @return   {return}       My Param doc
     */
    static cl::sycl::vector_class<cl::sycl::queue>
    init_queues (info::device_type device_ty)
    {
        std::vector<cl::sycl::queue> queues;
        for(auto d : device::get_devices(device_ty))
            queues.emplace_back(d);
        return queues;
    }
};

// Initialize the active_queue with the default queue
thread_local std::vector<cl::sycl::queue> QMgrHelper::active_queues
    = {default_selector()};

std::vector<cl::sycl::queue> QMgrHelper::cpu_queues
    = QMgrHelper::init_queues(info::device_type::cpu);

std::vector<cl::sycl::queue> QMgrHelper::gpu_queues
    = QMgrHelper::init_queues(info::device_type::gpu);

/*!
 *
 */
int64_t QMgrHelper::getCurrentQueue (void **QPtr)
{
    //std::cout << &QMgrHelper::active_queues << '\n';
    if(active_queues.empty())
        return error_reporter("No currently active queues.");
    auto last = QMgrHelper::active_queues.size() - 1;
    *QPtr = new queue(QMgrHelper::active_queues[last]);
    //std::cout << "(queue*)(*QPtr) :" << ((queue*)(*QPtr)) << '\n';

    return DPPL_SUCCESS;
}

/*!
 *
 */
int64_t
QMgrHelper::getQueue (void **Ptr2QPtr, sycl_device_type DeviceTy, size_t DNum)
{
    switch (DeviceTy)
    {
    case sycl_device_type::cpu:
    {
        try {
            *Ptr2QPtr = new queue(QMgrHelper::cpu_queues.at(DNum));
        }
        catch (const std::out_of_range& e) {
            std::stringstream ss;
            ss << "SYCL CPU device " << DNum << " not found on system.";
            return error_reporter(ss.str());
        }
    }
    case sycl_device_type::gpu:
    {
        try {
            *Ptr2QPtr = new queue(QMgrHelper::gpu_queues.at(DNum));
        }
        catch (const std::out_of_range& e) {
            std::stringstream ss;
            ss << "SYCL GPU device " << DNum << " not found on system.";
            return error_reporter(ss.str());
        }
    }
    default:
        return error_reporter("Unsupported device type.");
    }

    return DPPL_SUCCESS;
}

/*!
 *
 */
int64_t
QMgrHelper::setAsGlobalQueue (sycl_device_type DeviceTy, size_t DNum)
{
    if(active_queues.empty())
        return error_reporter("active queue vector is corrupted.");

    switch (DeviceTy)
    {
    case sycl_device_type::cpu:
    {
        try {
            active_queues[0] = cpu_queues.at(DNum);
        }
        catch (const std::out_of_range& e) {
            std::stringstream ss;
            ss << "SYCL CPU device " << DNum << " not found on system.";
            return error_reporter(ss.str());
        }
        break;
    }
    case sycl_device_type::gpu:
    {
        try {
            active_queues[0] = gpu_queues.at(DNum);
        }
        catch (const std::out_of_range& e) {
            std::stringstream ss;
            ss << "SYCL GPU device " << DNum << " not found on system.";
            return error_reporter(ss.str());
        }
        break;
    }
    default:
    {
        return error_reporter("Unsupported device type.");
    }
    }

    return DPPL_SUCCESS;
}

/*!
 *
 */
int64_t
QMgrHelper::setAsCurrentQueue (void **Ptr2QPtr,
                               dppl::sycl_device_type DeviceTy,
                               size_t DNum)
{
    if(active_queues.empty())
        return error_reporter("Why is there no previous global context?");

    switch (DeviceTy)
    {
    case sycl_device_type::cpu:
    {
        try {
            active_queues.emplace_back(cpu_queues.at(DNum));
            *Ptr2QPtr = new queue(active_queues[active_queues.size()-1]);
        }
        catch (const std::out_of_range& e) {
            std::stringstream ss;
            ss << "SYCL CPU device " << DNum << " not found on system.";
            return error_reporter(ss.str());
        }
        break;
    }
    case sycl_device_type::gpu:
    {
        try {
            active_queues.emplace_back(gpu_queues.at(DNum));
            *Ptr2QPtr = new queue(active_queues[active_queues.size()-1]);
        }
        catch (const std::out_of_range& e) {
            std::stringstream ss;
            ss << "SYCL GPU device " << DNum << " not found on system.";
            return error_reporter(ss.str());
        }
        break;
    }
    default:
    {
        return error_reporter("Unsupported device type.");
    }
    }

    return DPPL_SUCCESS;
}

/*!
 *
 */
int64_t
QMgrHelper::removeCurrentQueue ()
{
    if(active_queues.empty())
        return error_reporter("No active contexts");
    active_queues.pop_back();

    return DPPL_SUCCESS;
}

// This singleton function is needed to create the DpplOneAPIRuntimeHelper
// object in a predictable manner without which there is a chance of segfault.
// QMgrHelper& get_gRtHelper()
// {
//     static auto * helper = new QMgrHelper();
//     return *helper;
// }

// #define gRtHelper get_gRtHelper()

// thread_local std::vector<cl::sycl::queue> QMgrHelper::active_queues;


} /* end of anonymous namespace */


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// Free functions //////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
 *
 */
int64_t dppl::deleteQueue (void *Q)
{
    delete static_cast<queue*>(Q);
    return DPPL_SUCCESS;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////// DpplSyclQueueManager ////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
 *
 */
int64_t DpplSyclQueueManager::dump_queue (const void *QPtr) const
{
    auto Q = static_cast<const queue*>(QPtr);
    dump_device_info(Q->get_device());
    return DPPL_SUCCESS;
}

/*!
 *
 */
int64_t DpplSyclQueueManager::dump () const
{
    size_t i = 0;

    // Print out the info for each platform
    auto platforms = platform::get_platforms();
    for (auto &p : platforms) {
        std::cout << "---Platform " << i << '\n';
        dump_platform_info(p);
        ++i;
    }

    // Print out the info for CPU devices
    if (QMgrHelper::cpu_queues.size())
        std::cout << "---Number of available SYCL CPU queues: "
                  << QMgrHelper::cpu_queues.size() << '\n';
    else
        std::cout << "---No available SYCL CPU device\n";

    // Print out the info for GPU devices
    if (QMgrHelper::gpu_queues.size())
        std::cout << "---Number of available SYCL GPU queues: "
                  << QMgrHelper::gpu_queues.size() << '\n';
    else
        std::cout << "---No available SYCL GPU device\n";

    std::cout << "---Current queue :\n";
    dump_queue(&QMgrHelper::active_queues[0]);

    std::cout << "---Number of active queues : "
              << QMgrHelper::active_queues.size()
              << '\n';

    return DPPL_SUCCESS;
}

/*!
 *
 */
int64_t DpplSyclQueueManager::getNumPlatforms (size_t &platforms) const
{
    platforms = platform::get_platforms().size();
    return DPPL_SUCCESS;
}

/*!
 * Returns inside the passed in parameter the number of activated queues not
 * including the global queue that should always be activated.
 */
int64_t DpplSyclQueueManager::getNumActivatedQueues (size_t &numQueues) const
{
    if (QMgrHelper::active_queues.empty())
        return error_reporter("No active contexts");
    numQueues = QMgrHelper::active_queues.size() - 1;
    return DPPL_SUCCESS;
}

/*!
 *
 */
int64_t DpplSyclQueueManager::getCurrentQueue (void **Ptr2QPtr) const
{
    return QMgrHelper::getCurrentQueue(Ptr2QPtr);
}

/*!
 *
 */
int64_t DpplSyclQueueManager::getQueue (void **Ptr2QPtr,
                                        sycl_device_type DeviceTy,
                                        size_t DNum) const
{
    return QMgrHelper::getQueue(Ptr2QPtr, DeviceTy, DNum);
}

/*!
 * The function sets the global queue (i.e. the first queue in the
 * activeQueue vector) to the sycl::queue corresponding to the device
 * of given type and id. If no such device exists and the queue does not
 * exist, then DPPL_FAILURE is returned.
 */
int64_t DpplSyclQueueManager::setAsGlobalQueue (sycl_device_type DeviceTy,
                                                size_t DNum)
{
    return QMgrHelper::setAsGlobalQueue(DeviceTy, DNum);
}

/*!
 *
 */
int64_t
DpplSyclQueueManager::setAsCurrentQueue (void **Ptr2QPtr,
                                         sycl_device_type DeviceTy,
                                         size_t DNum)
{
    return QMgrHelper::setAsCurrentQueue(Ptr2QPtr, DeviceTy, DNum);
}

/*!
 *
 */
int64_t DpplSyclQueueManager::removeCurrentQueue ()
{
    return QMgrHelper::removeCurrentQueue();
}
