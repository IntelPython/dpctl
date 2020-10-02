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

    static QVec get_opencl_cpu_queues ()
    {
        QVec queues;

        for (auto &p : platform::get_platforms()) {
            auto Devices = p.get_devices();
            auto Ctx = context(Devices);
            if (p.is_host()) continue;
            for(auto &d : Devices) {
                auto devty = d.get_info<info::device::device_type>();
                auto be = p.get_backend();
                if(devty == info::device_type::cpu && be == backend::opencl) {
                    queues.emplace_back(Ctx, d);
                }
            }
        }

        return queues;
    }

    static QVec get_opencl_gpu_queues ()
    {
        QVec queues;

        for (auto &p : platform::get_platforms()) {
            auto Devices = p.get_devices();
            auto Ctx = context(Devices);
            if (p.is_host()) continue;
            for(auto &d : Devices) {
                auto devty = d.get_info<info::device::device_type>();
                auto be = p.get_backend();
                if(devty == info::device_type::gpu && be == backend::opencl) {
                    queues.emplace_back(Ctx, d);
                }
            }
        }

        return queues;
    }

    static QVec get_level0_gpu_queues ()
    {
        QVec queues;

        for (auto &p : platform::get_platforms()) {
            auto Devices = p.get_devices();
            auto Ctx = context(Devices);
            if (p.is_host()) continue;
            for(auto &d : Devices) {
                auto devty = d.get_info<info::device::device_type>();
                auto be = p.get_backend();
                if(devty == info::device_type::gpu &&
                   be == backend::level_zero) {
                    queues.emplace_back(Ctx, d);
                }
            }
        }

        return queues;
    }

    static QVec& get_active_queues ()
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

};

/*!
 * Allocates a new copy of the present top of stack queue, which can be the
 * default queue and returns to caller. The caller owns the pointer and is
 * responsible for deallocating it. The helper function DPPLQueue_Delete should
 * be used for that purpose.
 */
DPPLSyclQueueRef QMgrHelper::getCurrentQueue ()
{
    if(get_active_queues().empty()) {
        // \todo handle error
        std::cerr << "No currently active queues.\n";
        return nullptr;
    }
    auto last = QMgrHelper::get_active_queues().size() - 1;
    return wrap(new queue(QMgrHelper::get_active_queues()[last]));
}

/*!
 * Allocates a sycl::queue by copying from the cached {cpu|gpu}_queues vector
 * and returns it to the caller. The caller owns the pointer and is responsible
 * for deallocating it. The helper function DPPLQueue_Delete should
 * be used for that purpose.
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
        if (DNum >= get_opencl_cpu_queues().size()) {
            // \todo handle error
            std::cerr << "OpenCL CPU device " << DNum
                      << " not found on system.\n";
            return nullptr;
        }
        QRef = new queue(get_opencl_cpu_queues()[DNum]);
        break;
    }
    case DPPLSyclBEType::DPPL_OPENCL | DPPLSyclDeviceType::DPPL_GPU:
    {
        if (DNum >= get_opencl_gpu_queues().size()) {
            // \todo handle error
            std::cerr << "OpenCL GPU device " << DNum
                      << " not found on system.\n";
            return nullptr;
        }
        QRef = new queue(get_opencl_gpu_queues()[DNum]);
        break;
    }
    case DPPLSyclBEType::DPPL_LEVEL_ZERO | DPPLSyclDeviceType::DPPL_GPU:
    {
        if (DNum >= get_level0_gpu_queues().size()) {
            // \todo handle error
            std::cerr << "Level-0 GPU device " << DNum
                      << " not found on system.\n";
            return nullptr;
        }
        QRef = new queue(get_level0_gpu_queues()[DNum]);
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
    if(get_active_queues().empty()) {
        std::cerr << "active queue vector is corrupted.\n";
        return;
    }

    switch (BETy|DeviceTy)
    {
    case DPPLSyclBEType::DPPL_OPENCL | DPPLSyclDeviceType::DPPL_CPU:
    {
        if (DNum >= get_opencl_cpu_queues().size()) {
            // \todo handle error
            std::cerr << "OpenCL CPU device " << DNum
                      << " not found on system\n.";
            break;
        }
        get_active_queues()[0] = get_opencl_cpu_queues()[DNum];
        break;
    }
   case DPPLSyclBEType::DPPL_OPENCL | DPPLSyclDeviceType::DPPL_GPU:
    {
        if (DNum >= get_opencl_gpu_queues().size()) {
            // \todo handle error
            std::cerr << "OpenCL GPU device " << DNum
                      << " not found on system\n.";
            break;
        }
        get_active_queues()[0] = get_opencl_gpu_queues()[DNum];
        break;
    }
    case DPPLSyclBEType::DPPL_LEVEL_ZERO | DPPLSyclDeviceType::DPPL_GPU:
    {
        if (DNum >= get_level0_gpu_queues().size()) {
            // \todo handle error
            std::cerr << "Level-0 GPU device " << DNum
                      << " not found on system\n.";
            break;
        }
        get_active_queues()[0] = get_level0_gpu_queues()[DNum];
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
    if(get_active_queues().empty()) {
        std::cerr << "Why is there no previous global context?\n";
        return nullptr;
    }

    switch (BETy|DeviceTy)
    {
    case DPPLSyclBEType::DPPL_OPENCL | DPPLSyclDeviceType::DPPL_CPU:
    {
        if (DNum >= get_opencl_cpu_queues().size()) {
            // \todo handle error
            std::cerr << "OpenCL CPU device " << DNum
                      << " not found on system\n.";
            return nullptr;
        }
        get_active_queues().emplace_back(get_opencl_cpu_queues()[DNum]);
        QRef = new queue(get_active_queues()[get_active_queues().size()-1]);
        break;
    }
    case DPPLSyclBEType::DPPL_OPENCL | DPPLSyclDeviceType::DPPL_GPU:
    {
        if (DNum >= get_opencl_gpu_queues().size()) {
            // \todo handle error
            std::cerr << "OpenCL GPU device " << DNum
                      << " not found on system\n.";
            return nullptr;
        }
        get_active_queues().emplace_back(get_opencl_gpu_queues()[DNum]);
        QRef = new queue(get_active_queues()[get_active_queues().size()-1]);
        break;
    }
    case DPPLSyclBEType::DPPL_LEVEL_ZERO | DPPLSyclDeviceType::DPPL_GPU:
    {
        if (DNum >= get_level0_gpu_queues().size()) {
            // \todo handle error
            std::cerr << "Level-0 GPU device " << DNum
                      << " not found on system\n.";
            return nullptr;
        }
        get_active_queues().emplace_back(get_level0_gpu_queues()[DNum]);
        QRef = new queue(get_active_queues()[get_active_queues().size()-1]);
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
    if(get_active_queues().size() <= 1 ) {
        std::cerr << "No active contexts.\n";
        return;
    }
    get_active_queues().pop_back();
}

} /* end of anonymous namespace */

//----------------------------- Public API -----------------------------------//

/*!
 * Returns inside the number of activated queues not including the global queue
 * (QMgrHelper::active_queues[0]).
 */
size_t DPPLQueueMgr_GetNumActivatedQueues ()
{
    if (QMgrHelper::get_active_queues().empty()) {
        // \todo handle error
        std::cerr << "No active contexts.\n";
        return 0;
    }
    return QMgrHelper::get_active_queues().size() - 1;
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
        return QMgrHelper::get_opencl_cpu_queues().size();
    }
    case DPPLSyclBEType::DPPL_OPENCL | DPPLSyclDeviceType::DPPL_GPU:
    {
        return QMgrHelper::get_opencl_gpu_queues().size();
    }
    case DPPLSyclBEType::DPPL_LEVEL_ZERO | DPPLSyclDeviceType::DPPL_GPU:
    {
        return QMgrHelper::get_level0_gpu_queues().size();
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
