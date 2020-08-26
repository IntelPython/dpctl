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
#include "dppl_sycl_queue_interface.h"
#include "Support/CBindingWrapping.h"
#include <cassert>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <CL/sycl.hpp>                /* SYCL headers   */

using namespace cl::sycl;

/*------------------------------- Private helpers ----------------------------*/

// Anonymous namespace for private helpers
namespace
{

 // Create wrappers for C Binding types (see CBindingWrapping.h).
 DEFINE_SIMPLE_CONVERSION_FUNCTIONS(queue, DPPLSyclQueueRef)

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

void error_reporter (const std::string & msg)
{
    throw std::runtime_error("Error: " + msg);
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

    static __dppl_give DPPLSyclQueueRef
    getQueue (DPPLSyclDeviceType DeviceTy, size_t DNum);

    static __dppl_give DPPLSyclQueueRef
    getCurrentQueue ();

    static void
    setAsDefaultQueue (DPPLSyclDeviceType DeviceTy, size_t DNum);

    static __dppl_give DPPLSyclQueueRef
    pushSyclQueue (DPPLSyclDeviceType DeviceTy, size_t DNum);

    static void
    popSyclQueue ();

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
DPPLSyclQueueRef QMgrHelper::getCurrentQueue ()
{
    if(active_queues.empty())
        error_reporter("No currently active queues.");
    auto last = QMgrHelper::active_queues.size() - 1;
    return wrap(new queue(QMgrHelper::active_queues[last]));
}

/*!
 * Allocates a sycl::queue by copying from the cached {cpu|gpu}_queues vector
 * and returns it to the caller. The caller owns the pointer and is responsible
 * for deallocating it. The helper function deleteQueue can be used is for that
 * purpose.
 */
DPPLSyclQueueRef
QMgrHelper::getQueue (DPPLSyclDeviceType DeviceTy,
                      size_t DNum)
{
    queue *QRef = nullptr;

    switch (DeviceTy)
    {
    case DPPLSyclDeviceType::DPPL_CPU:
    {
        if (DNum >= cpu_queues.size()) {
            std::stringstream ss;
            ss << "SYCL CPU device " << DNum << " not found on system.";
            error_reporter(ss.str());
        }
        QRef = new queue(QMgrHelper::cpu_queues[DNum]);
        break;
    }
    case DPPLSyclDeviceType::DPPL_GPU:
    {
        if (DNum >= gpu_queues.size()) {
            std::stringstream ss;
            ss << "SYCL GPU device " << DNum << " not found on system.";
            error_reporter(ss.str());
        }
        QRef = new queue(QMgrHelper::gpu_queues[DNum]);
        break;
    }
    default:
        error_reporter("Unsupported device type.");
    }

    return wrap(QRef);
}

/*!
 * Changes the first entry into the stack, i.e., the default queue to a new
 * sycl::queue corresponding to the device type and device number.
 */
void
QMgrHelper::setAsDefaultQueue (DPPLSyclDeviceType DeviceTy, size_t DNum)
{
    if(active_queues.empty())
        error_reporter("active queue vector is corrupted.");

    switch (DeviceTy)
    {
    case DPPLSyclDeviceType::DPPL_CPU:
    {
        if (DNum >= cpu_queues.size()) {
            std::stringstream ss;
            ss << "SYCL CPU device " << DNum << " not found on system.";
            error_reporter(ss.str());
        }
        active_queues[0] = cpu_queues[DNum];
        break;
    }
    case DPPLSyclDeviceType::DPPL_GPU:
    {
        if (DNum >= gpu_queues.size()) {
            std::stringstream ss;
            ss << "SYCL GPU device " << DNum << " not found on system.";
            error_reporter(ss.str());
        }
        active_queues[0] = gpu_queues[DNum];
        break;
    }
    default:
    {
        error_reporter("Unsupported device type.");
    }
    }
}

/*!
 * Allocates a new sycl::queue by copying from the cached {cpu|gpu}_queues
 * vector. The pointer returned is now owned by the caller and must be properly
 * cleaned up. The helper function DPPLDeleteQueue() can be used is for that
 * purpose.
 */
DPPLSyclQueueRef
QMgrHelper::pushSyclQueue (DPPLSyclDeviceType DeviceTy, size_t DNum)
{
    queue *QRef = nullptr;
    if(active_queues.empty())
        error_reporter("Why is there no previous global context?");

    switch (DeviceTy)
    {
    case DPPLSyclDeviceType::DPPL_CPU:
    {
        if (DNum >= cpu_queues.size()) {
            std::stringstream ss;
            ss << "SYCL CPU device " << DNum << " not found on system.";
            error_reporter(ss.str());
        }
        active_queues.emplace_back(cpu_queues[DNum]);
        QRef = new queue(active_queues[active_queues.size()-1]);
        break;
    }
    case DPPLSyclDeviceType::DPPL_GPU:
    {
        if (DNum >= gpu_queues.size()) {
            std::stringstream ss;
            ss << "SYCL GPU device " << DNum << " not found on system.";
            error_reporter(ss.str());
        }
        active_queues.emplace_back(gpu_queues[DNum]);
        QRef = new queue(active_queues[active_queues.size()-1]);
        break;
    }
    default:
    {
        error_reporter("Unsupported device type.");
    }
    }

    return wrap(QRef);
}

/*!
 * If there were any sycl::queues that were activated and added to the stack of
 * activated queues then the top of the stack entry is popped. Note that since
 * the same std::vector is used to keep track of the activated queues and the
 * global queue a popSyclQueue call can never make the stack empty. Even
 * after all activated queues are popped, the global queue is still available as
 * the first element added to the stack.
 */
void
QMgrHelper::popSyclQueue ()
{
    // The first queue which is the "default" queue can not be removed.
    if(active_queues.size() <= 1 )
        error_reporter("No active contexts");
    active_queues.pop_back();
}

} /* end of anonymous namespace */


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// Free functions //////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
 * Delete the passed in pointer after verifying it points to a sycl::queue.
 */
void DPPLDeleteQueue (DPPLSyclQueueRef QRef)
{
    delete unwrap(QRef);
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////// DpplSyclQueueManager ////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
 * Prints some of the device info metadata for the specified sycl::queue.
 * Currently, device name, driver version, device vendor, and device profile
 * are printed out. More attributed may be added later.
 */
void DPPLDumpDeviceInfo (const DPPLSyclQueueRef QRef)
{
    auto Q = unwrap(QRef);
    dump_device_info(Q->get_device());
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
void DPPLDumpPlatformInfo ()
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
    DPPLDumpDeviceInfo(wrap(&QMgrHelper::active_queues[0]));

    std::cout << "---Number of active queues : "
              << QMgrHelper::active_queues.size()
              << '\n';
}

/*!
 * Returns inside the platform param the number of SYCL platforms on the system.
 */
size_t DPPLGetNumPlatforms ()
{
    return platform::get_platforms().size();
}

/*!
 * Returns inside the numQueues param the number of activated queues not
 * including the global queue that should always be activated.
 */
size_t DPPLGetNumActivatedQueues ()
{
    if (QMgrHelper::active_queues.empty())
        error_reporter("No active contexts");
    return QMgrHelper::active_queues.size() - 1;
}

/*!
 * Returns the number of CPU queues.
 */
size_t DPPLGetNumCPUQueues ()
{
    return QMgrHelper::cpu_queues.size();
}

/*!
 * Returns the number of GPU queues.
 */
size_t DPPLGetNumGPUQueues ()
{
    return QMgrHelper::gpu_queues.size();
}

/*!
 * Returns a copy of the current queue inside the Ptr2QPtr param.
 */
DPPLSyclQueueRef DPPLGetCurrentQueue ()
{
    return QMgrHelper::getCurrentQueue();
}

/*!
 * Returns inside the Ptr2QPtr param a copy of a sycl::queue corresponding to
 * the specified device type and device number.
 */
DPPLSyclQueueRef DPPLGetQueue (DPPLSyclDeviceType DeviceTy,
                               size_t DNum)
{
    return QMgrHelper::getQueue(DeviceTy, DNum);
}

/*!
 * The function sets the global queue (i.e. the first queue in the
 * activeQueue vector) to the sycl::queue corresponding to the device
 * of given type and id. If no such device exists and the queue does not
 * exist, then DPPL_FAILURE is returned.
 */
void DPPLSetAsDefaultQueue (DPPLSyclDeviceType DeviceTy, size_t DNum)
{
    QMgrHelper::setAsDefaultQueue(DeviceTy, DNum);
}

/*!
 * Pushes a new sycl::queue to the stack of activated queues. A copy of the
 * queue is returned to the caller inside the Ptr2QPtr param.
 */
__dppl_give DPPLSyclQueueRef DPPLPushSyclQueue (DPPLSyclDeviceType DeviceTy,
                                                size_t DNum)
{
    return QMgrHelper::pushSyclQueue(DeviceTy, DNum);
}

/*!
 * Pops the top of stack element for the stack of currently activated
 * sycl::queues. Returns DPPL_ERROR if the stack has no activated queues other
 * than the default global queue.
 */
void DPPLPopSyclQueue ()
{
    QMgrHelper::popSyclQueue();
}
