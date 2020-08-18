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
 * @brief A helper class to support the DPPLSyclQueuemanager.
 *
 * The QMgrHelper is needed so that sycl headers are not exposed at the
 * top-level DPPL API.
 *
 */
class QMgrHelper
{
public:
    static std::vector<cl::sycl::queue>   cpu_queues;
    static std::vector<cl::sycl::queue>   gpu_queues;
    static thread_local std::vector<cl::sycl::queue> active_queues;

    static int64_t
    getQueue (void **QPtr, sycl_device_type DeviceTy, size_t DNum);

    static int64_t
    getCurrentQueue (void **q);

    static int64_t
    setAsDefaultQueue (sycl_device_type DeviceTy, size_t DNum);

    static int64_t
    setAsCurrentQueue (void **QPtr,
                       dppl::sycl_device_type DeviceTy,
                       size_t DNum);

    static int64_t
    removeCurrentQueue ();

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
 * Allocates a new copy of the present top of stack queue, which can be the
 * default queue and returns to caller. The caller owns the pointer and is
 * responsible for deallocating it. The helper function deleteQueue can be used
 * is for that purpose.
 */
int64_t QMgrHelper::getCurrentQueue (void **QPtr)
{
    if(active_queues.empty())
        return error_reporter("No currently active queues.");
    auto last = QMgrHelper::active_queues.size() - 1;
    *QPtr = new queue(QMgrHelper::active_queues[last]);

    return DPPL_SUCCESS;
}

/*!
 * Allocates a sycl::queue by copying from the cached {cpu|gpu}_queues vector
 * and returns it to the caller. The caller owns the pointer and is responsible
 * for deallocating it. The helper function deleteQueue can be used is for that
 * purpose.
 */
int64_t
QMgrHelper::getQueue (void **Ptr2QPtr, sycl_device_type DeviceTy, size_t DNum)
{
    switch (DeviceTy)
    {
    case sycl_device_type::DPPL_CPU:
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
    case sycl_device_type::DPPL_GPU:
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
 * Changes the first entry into the stack, i.e., the default queue to a new
 * sycl::queue corresponding to the device type and device number.
 */
int64_t
QMgrHelper::setAsDefaultQueue (sycl_device_type DeviceTy, size_t DNum)
{
    if(active_queues.empty())
        return error_reporter("active queue vector is corrupted.");

    switch (DeviceTy)
    {
    case sycl_device_type::DPPL_CPU:
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
    case sycl_device_type::DPPL_GPU:
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
 * Allocates a new sycl::queue by copying from the cached {cpu|gpu}_queues
 * vector. The pointer returned is now owned by the caller and must be properly
 * cleaned up. The helper function deleteQueue can be used is for that purpose.
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
    case sycl_device_type::DPPL_CPU:
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
    case sycl_device_type::DPPL_GPU:
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
 * If there were any sycl::queues that were activated and added to the stack of
 * activated queues then the top of the stack entry is popped. Note that since
 * the same std::vector is used to keep track of the activated queues and the
 * global queue a removeCurrentQueue call can never make the stack empty. Even
 * after all activated queues are popped, the global queue is still available as
 * the first element added to the stack.
 */
int64_t
QMgrHelper::removeCurrentQueue ()
{
    // The first queue which is the "default" queue can not be removed.
    if(active_queues.size() <= 1 )
        return error_reporter("No active contexts");
    active_queues.pop_back();

    return DPPL_SUCCESS;
}

} /* end of anonymous namespace */


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// Free functions //////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
 * Delete the passed in pointer after verifying it points to a sycl::queue.
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
 * Prints some of the device info metadata for the specified sycl::queue.
 * Currently, device name, driver version, device vendor, and device profile
 * are printed out. More attributed may be added later.
 */
int64_t DpplSyclQueueManager::dumpDeviceInfo (const void *QPtr) const
{
    auto Q = static_cast<const queue*>(QPtr);
    dump_device_info(Q->get_device());
    return DPPL_SUCCESS;
}

/*!
 * Prints out number of available Sycl platforms, number of CPU queues, number
 * of GPU queues, metadata about the current global queue, and how many queues
 * are currently activated. More information can be added in future, and
 * functions to extract these information using Sycl API (e.g. device_info)
 * may also be added. For now, this function can be used as a basic canary test
 * to check if the queue manager was properly initialized.
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
    dumpDeviceInfo(&QMgrHelper::active_queues[0]);

    std::cout << "---Number of active queues : "
              << QMgrHelper::active_queues.size()
              << '\n';

    return DPPL_SUCCESS;
}

/*!
 * Returns inside the platform param the number of Sycl platforms on the system.
 */
int64_t DpplSyclQueueManager::getNumPlatforms (size_t &platforms) const
{
    platforms = platform::get_platforms().size();
    return DPPL_SUCCESS;
}

/*!
 * Returns inside the numQueues param the number of activated queues not
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
 * Returns the number of CPU queues.
 */
int64_t DpplSyclQueueManager::getNumCPUQueues (size_t &numQueues) const
{
    numQueues = QMgrHelper::cpu_queues.size();
    return DPPL_SUCCESS;
}

/*!
 * Returns the number of GPU queues.
 */
int64_t DpplSyclQueueManager::getNumGPUQueues (size_t &numQueues) const
{
    numQueues = QMgrHelper::gpu_queues.size();
    return DPPL_SUCCESS;
}

/*!
 * Returns a copy of the current queue inside the Ptr2QPtr param.
 */
int64_t DpplSyclQueueManager::getCurrentQueue (void **Ptr2QPtr) const
{
    return QMgrHelper::getCurrentQueue(Ptr2QPtr);
}

/*!
 * Returns inside the Ptr2QPtr param a copy of a sycl::queue corresponding to
 * the specified device type and device number.
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
int64_t DpplSyclQueueManager::setAsDefaultQueue (sycl_device_type DeviceTy,
                                                 size_t DNum)
{
    return QMgrHelper::setAsDefaultQueue(DeviceTy, DNum);
}

/*!
 * Pushes a new sycl::queue to the stack of activated queues. A copy of the
 * queue is returned to the caller inside the Ptr2QPtr param.
 */
int64_t
DpplSyclQueueManager::setAsCurrentQueue (void **Ptr2QPtr,
                                         sycl_device_type DeviceTy,
                                         size_t DNum)
{
    return QMgrHelper::setAsCurrentQueue(Ptr2QPtr, DeviceTy, DNum);
}

/*!
 * Pops the top of stack element for the stack of currently activated
 * sycl::queues. Returns DPPL_ERROR if the stack has no activated queues other
 * than the default global queue.
 */
int64_t DpplSyclQueueManager::removeCurrentQueue ()
{
    return QMgrHelper::removeCurrentQueue();
}
