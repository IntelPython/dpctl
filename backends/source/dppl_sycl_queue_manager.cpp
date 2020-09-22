//===--- dppl_sycl_queue_manager.cpp - DPPL-SYCL interface --*- C++ -*---===//
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
/// dppl_sycl_queue_manager.h.
///
//===----------------------------------------------------------------------===//
#include "dppl_sycl_queue_manager.h"
#include "Support/CBindingWrapping.h"
#include <exception>
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
    static std::vector<cl::sycl::queue>&
    cpu_queues_ ()
    {
        static std::vector<cl::sycl::queue>* cpu_queues =
            QMgrHelper::init_queues(info::device_type::cpu);
        return *cpu_queues;
    }

    static std::vector<cl::sycl::queue>&
    gpu_queues_ ()
    {
        static std::vector<cl::sycl::queue>* gpu_queues =
            QMgrHelper::init_queues(info::device_type::gpu);
        return *gpu_queues;
    }

    static std::vector<cl::sycl::queue>&
    active_queues_ ()
    {
        thread_local static std::vector<cl::sycl::queue>* active_queues =
            new std::vector<cl::sycl::queue>({default_selector()});
        return *active_queues;
    }

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

    static cl::sycl::vector_class<cl::sycl::queue>*
    init_queues (info::device_type device_ty)
    {
        auto queues = new std::vector<cl::sycl::queue>();
        for(auto d : device::get_devices(device_ty))
            queues->emplace_back(d);
        return queues;
    }
};

// make function call like access to variable
// it is for minimizing code changes during replacing static vars with functions
// it could be refactored by replacing variable with function call
// scope of this variables is only this file
#define cpu_queues    cpu_queues_()
#define gpu_queues    gpu_queues_()
#define active_queues active_queues_()


//----------------------------- Public API -----------------------------------//

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
QMgrHelper::getQueue (DPPLSyclDeviceType DeviceTy, size_t DNum)
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
 * cleaned up. The helper function DPPLDeleteSyclQueue() can be used is for that
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
 * If there were any sycl::queue that were activated and added to the stack of
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

/*!
 * Returns inside the number of activated queues not including the global queue
 * (QMgrHelper::active_queues[0]).
 */
size_t DPPLQueueMgr_GetNumActivatedQueues ()
{
    if (QMgrHelper::active_queues.empty())
        error_reporter("No active contexts");
    return QMgrHelper::active_queues.size() - 1;
}

/*!
 * Returns the number of CPU queues.
 */
size_t DPPLQueueMgr_GetNumCPUQueues ()
{
    return QMgrHelper::cpu_queues.size();
}

/*!
 * Returns the number of GPU queues.
 */
size_t DPPLQueueMgr_GetNumGPUQueues ()
{
    return QMgrHelper::gpu_queues.size();
}

/*!
 * \see QMgrHelper::getCurrentQueue()
 */
DPPLSyclQueueRef DPPLQueueMgr_GetCurrentQueue ()
{
    return QMgrHelper::getCurrentQueue();
}

/*!
 * Returns a copy of a sycl::queue corresponding to the specified device type
 * and device number. A runtime_error gets thrown if no such device exists.
 */
DPPLSyclQueueRef DPPLQueueMgr_GetQueue (DPPLSyclDeviceType DeviceTy,
                                        size_t DNum)
{
    return QMgrHelper::getQueue(DeviceTy, DNum);
}

/*!
 * The function sets the global queue, i.e., the sycl::queue object at
 * QMgrHelper::active_queues[0] vector to the sycl::queue corresponding to the
 * specified device type and id. A runtime_error gets thrown if no such device
 * exists.
 */
void DPPLQueueMgr_SetAsDefaultQueue (DPPLSyclDeviceType DeviceTy, size_t DNum)
{
    QMgrHelper::setAsDefaultQueue(DeviceTy, DNum);
}

/*!
 * \see QMgrHelper::pushSyclQueue()
 */
__dppl_give DPPLSyclQueueRef
DPPLQueueMgr_PushQueue (DPPLSyclDeviceType DeviceTy, size_t DNum)
{
    return QMgrHelper::pushSyclQueue(DeviceTy, DNum);
}

/*!
 * \see QMgrHelper::popSyclQueue()
 */
void DPPLQueueMgr_PopQueue ()
{
    QMgrHelper::popSyclQueue();
}
