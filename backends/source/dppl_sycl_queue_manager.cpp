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
    using QVec = vector_class<queue>;

    static QVec& opencl_cpu_queues_ ()
    {
        static QVec* cpu_queues = init_queues(info::device_type::cpu);
        return *cpu_queues;
    }

    static QVec& opencl_gpu_queues_ ()
    {
        static QVec* gpu_queues = init_queues(info::device_type::gpu);
        return *gpu_queues;
    }

    static QVec& level0_gpu_queues_ ()
    {
        static QVec* gpu_queues = init_queues(info::device_type::gpu);
        return *gpu_queues;
    }

    static QVec& active_queues_ ()
    {
        thread_local static QVec* active_queues =
            new QVec({default_selector()});
        return *active_queues;
    }

    static __dppl_give DPPLSyclQueueRef
    getQueue (DPPLSyclBEType BETy,
              DPPLSyclDeviceType DeviceTy,
              size_t DNum);

    static __dppl_give DPPLSyclQueueRef
    getCurrentQueue ();

    static void
    setAsDefaultQueue (DPPLSyclBEType BETy,
                       DPPLSyclDeviceType DeviceTy,
                       size_t DNum);

    static __dppl_give DPPLSyclQueueRef
    pushSyclQueue (DPPLSyclBEType BETy,
                   DPPLSyclDeviceType DeviceTy,
                   size_t DNum);

    static void
    popSyclQueue ();

    static QVec* init_queues (info::device_type device_ty)
    {

        auto queues = new QVec();
        for(auto d : device::get_devices(device_ty))
            queues->emplace_back(d);
        return queues;
    }
};

// make function call like access to variable
// it is for minimizing code changes during replacing static vars with functions
// it could be refactored by replacing variable with function call
// scope of this variables is only this file
#define opencl_cpu_queues opencl_cpu_queues_()
#define opencl_gpu_queues opencl_gpu_queues_()
#define level0_gpu_queues level0_gpu_queues_()
#define active_queues     active_queues_()


/*!
 * Allocates a new copy of the present top of stack queue, which can be the
 * default queue and returns to caller. The caller owns the pointer and is
 * responsible for deallocating it. The helper function deleteQueue can be used
 * is for that purpose.
 */
DPPLSyclQueueRef QMgrHelper::getCurrentQueue ()
{
    if(active_queues.empty()) {
        // \todo handle error
        std::cerr << "No currently active queues.\n";
        return nullptr;
    }
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
QMgrHelper::getQueue (DPPLSyclBEType BETy,
                      DPPLSyclDeviceType DeviceTy,
                      size_t DNum)
{
    queue *QRef = nullptr;

    switch (BETy|DeviceTy)
    {
    case DPPLSyclBEType::DPPL_OPENCL | DPPLSyclDeviceType::DPPL_CPU:
    {
        if (DNum >= opencl_cpu_queues.size()) {
            // \todo handle error
            std::cerr << "OpenCL CPU device " << DNum
                      << " not found on system.\n";
            return nullptr;
        }
        QRef = new queue(opencl_cpu_queues[DNum]);
        break;
    }
    case DPPLSyclBEType::DPPL_OPENCL | DPPLSyclDeviceType::DPPL_GPU:
    {
        if (DNum >= opencl_gpu_queues.size()) {
            // \todo handle error
            std::cerr << "OpenCL GPU device " << DNum
                      << " not found on system.\n";
            return nullptr;
        }
        QRef = new queue(opencl_gpu_queues[DNum]);
        break;
    }
    case DPPLSyclBEType::DPPL_LEVEL_ZERO | DPPLSyclDeviceType::DPPL_GPU:
    {
        if (DNum >= level0_gpu_queues.size()) {
            // \todo handle error
            std::cerr << "Level-0 GPU device " << DNum
                      << " not found on system.\n";
            return nullptr;
        }
        QRef = new queue(level0_gpu_queues[DNum]);
        break;
    }
    default:
        std::cerr << "Unsupported device type.\n";
        return nullptr;
    }

    return wrap(QRef);
}

/*!
 * Changes the first entry into the stack, i.e., the default queue to a new
 * sycl::queue corresponding to the device type and device number.
 */
void
QMgrHelper::setAsDefaultQueue (DPPLSyclBEType BETy,
                               DPPLSyclDeviceType DeviceTy,
                               size_t DNum)
{
    if(active_queues.empty()) {
        std::cerr << "active queue vector is corrupted.\n";
        return;
    }

    switch (BETy|DeviceTy)
    {
    case DPPLSyclBEType::DPPL_OPENCL | DPPLSyclDeviceType::DPPL_CPU:
    {
        if (DNum >= opencl_cpu_queues.size()) {
            // \todo handle error
            std::cerr << "OpenCL CPU device " << DNum
                      << " not found on system\n.";
            break;
        }
        active_queues[0] = opencl_cpu_queues[DNum];
        break;
    }
   case DPPLSyclBEType::DPPL_OPENCL | DPPLSyclDeviceType::DPPL_GPU:
    {
        if (DNum >= opencl_gpu_queues.size()) {
            // \todo handle error
            std::cerr << "OpenCL GPU device " << DNum
                      << " not found on system\n.";
            break;
        }
        active_queues[0] = opencl_gpu_queues[DNum];
        break;
    }
    case DPPLSyclBEType::DPPL_LEVEL_ZERO | DPPLSyclDeviceType::DPPL_GPU:
    {
        if (DNum >= level0_gpu_queues.size()) {
            // \todo handle error
            std::cerr << "Level-0 GPU device " << DNum
                      << " not found on system\n.";
            break;
        }
        active_queues[0] = level0_gpu_queues[DNum];
        break;
    }
    default:
    {
        std::cerr << "Unsupported device type.\n";
        return;
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
QMgrHelper::pushSyclQueue (DPPLSyclBEType BETy,
                           DPPLSyclDeviceType DeviceTy,
                           size_t DNum)
{
    queue *QRef = nullptr;
    if(active_queues.empty()) {
        std::cerr << "Why is there no previous global context?\n";
        return nullptr;
    }

    switch (BETy|DeviceTy)
    {
    case DPPLSyclBEType::DPPL_OPENCL | DPPLSyclDeviceType::DPPL_CPU:
    {
        if (DNum >= opencl_cpu_queues.size()) {
            // \todo handle error
            std::cerr << "OpenCL CPU device " << DNum
                      << " not found on system\n.";
            return nullptr;
        }
        active_queues.emplace_back(opencl_cpu_queues[DNum]);
        QRef = new queue(active_queues[active_queues.size()-1]);
        break;
    }
    case DPPLSyclBEType::DPPL_OPENCL | DPPLSyclDeviceType::DPPL_GPU:
    {
        if (DNum >= opencl_gpu_queues.size()) {
            // \todo handle error
            std::cerr << "OpenCL GPU device " << DNum
                      << " not found on system\n.";
            return nullptr;
        }
        active_queues.emplace_back(opencl_gpu_queues[DNum]);
        QRef = new queue(active_queues[active_queues.size()-1]);
        break;
    }
    case DPPLSyclBEType::DPPL_LEVEL_ZERO | DPPLSyclDeviceType::DPPL_GPU:
    {
        if (DNum >= level0_gpu_queues.size()) {
            // \todo handle error
            std::cerr << "Level-0 GPU device " << DNum
                      << " not found on system\n.";
            return nullptr;
        }
        active_queues.emplace_back(level0_gpu_queues[DNum]);
        QRef = new queue(active_queues[active_queues.size()-1]);
        break;
    }
    default:
    {
        std::cerr << "Unsupported device type.\n";
        return nullptr;
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
    if(active_queues.size() <= 1 ) {
        std::cerr << "No active contexts.\n";
        return;
    }
    active_queues.pop_back();
}

} /* end of anonymous namespace */

//----------------------------- Public API -----------------------------------//

/*!
 * Returns inside the number of activated queues not including the global queue
 * (QMgrHelper::active_queues[0]).
 */
size_t DPPLQueueMgr_GetNumActivatedQueues ()
{
    if (QMgrHelper::active_queues.empty()) {
        // \todo handle error
        std::cerr << "No active contexts.\n";
        return 0;
    }
    return QMgrHelper::active_queues.size() - 1;
}

/*!
 * Returns the number of available queues for a specific backend and device
 * type combination.
 */
size_t DPPLQueueMgr_GetNumQueues (DPPLSyclBEType BETy,
                                  DPPLSyclDeviceType DeviceTy)
{
    switch (BETy|DeviceTy)
    {
    case DPPLSyclBEType::DPPL_OPENCL | DPPLSyclDeviceType::DPPL_CPU:
    {
        return QMgrHelper::opencl_cpu_queues.size();
    }
    case DPPLSyclBEType::DPPL_OPENCL | DPPLSyclDeviceType::DPPL_GPU:
    {
        return QMgrHelper::opencl_gpu_queues.size();
    }
    case DPPLSyclBEType::DPPL_LEVEL_ZERO | DPPLSyclDeviceType::DPPL_GPU:
    {
        return QMgrHelper::level0_gpu_queues.size();
    }
    default:
    {
        // \todo handle error
        std::cerr << "Unsupported device type.\n";
        return 0;
    }
    }
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
DPPLSyclQueueRef DPPLQueueMgr_GetQueue (DPPLSyclBEType BETy,
                                        DPPLSyclDeviceType DeviceTy,
                                        size_t DNum)
{
    return QMgrHelper::getQueue(BETy, DeviceTy, DNum);
}

/*!
 * The function sets the global queue, i.e., the sycl::queue object at
 * QMgrHelper::active_queues[0] vector to the sycl::queue corresponding to the
 * specified device type and id. A runtime_error gets thrown if no such device
 * exists.
 */
void DPPLQueueMgr_SetAsDefaultQueue (DPPLSyclBEType BETy,
                                     DPPLSyclDeviceType DeviceTy,
                                     size_t DNum)
{
    QMgrHelper::setAsDefaultQueue(BETy, DeviceTy, DNum);
}

/*!
 * \see QMgrHelper::pushSyclQueue()
 */
__dppl_give DPPLSyclQueueRef
DPPLQueueMgr_PushQueue (DPPLSyclBEType BETy,
                        DPPLSyclDeviceType DeviceTy,
                        size_t DNum)
{
    return QMgrHelper::pushSyclQueue(BETy, DeviceTy, DNum);
}

/*!
 * \see QMgrHelper::popSyclQueue()
 */
void DPPLQueueMgr_PopQueue ()
{
    QMgrHelper::popSyclQueue();
}
