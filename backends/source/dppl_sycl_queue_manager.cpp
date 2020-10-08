//===--------- dppl_sycl_queue_manager.cpp - dpctl-C_API  --*-- C++ ---*---===//
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
/// This file implements the data types and functions declared in
/// dppl_sycl_queue_manager.h.
///
//===----------------------------------------------------------------------===//
#include "dppl_sycl_queue_manager.h"
#include "Support/CBindingWrapping.h"
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

    static QVec* init_queues (backend BE, info::device_type DTy)
    {
        QVec *queues = new QVec();
        auto Platforms = platform::get_platforms();
        for (auto &p : Platforms) {
            if (p.is_host()) continue;
            auto be = p.get_backend();
            auto Devices = p.get_devices();

            if (Devices.size() == 1) {
                auto d = Devices[0];
                auto devty = d.get_info<info::device::device_type>();
                if(devty == DTy && be == BE) {
                    auto Ctx = context(d);
                    queues->emplace_back(Ctx, d);
                    break;
                }
            }
            else {
                vector_class<device> SelectedDevices;
                for(auto &d : Devices) {
                    auto devty = d.get_info<info::device::device_type>();
                    if(devty == DTy && be == BE)
                        SelectedDevices.push_back(d);
                }
                if (SelectedDevices.size() > 0) {
                    auto Ctx = context(SelectedDevices);
                    auto d = SelectedDevices[0];
                    queues->emplace_back(Ctx, d);
                }
            }
        }
        return queues;
    }

    static QVec* init_active_queues ()
    {
        QVec *active_queues;
        try {
            auto def_device = std::move(default_selector().select_device());
            auto BE = def_device.get_platform().get_backend();
            auto DevTy = def_device.get_info<info::device::device_type>();

            // \todo : We need to have a better way to match the default device
            // to what SYCL returns based on the same scoring logic. Just
            // storing the first device is not correct when we will have
            // multiple devices of same type.
            if(BE == backend::opencl &&
                DevTy == info::device_type::cpu) {
                active_queues = new QVec({get_opencl_cpu_queues()[0]});
            }
            else if(BE == backend::opencl &&
                DevTy == info::device_type::gpu) {
                active_queues = new QVec({get_opencl_gpu_queues()[0]});
            }
            else if(BE == backend::level_zero &&
                DevTy == info::device_type::gpu) {
                active_queues =  new QVec({get_level0_gpu_queues()[0]});
            }
            else {
                active_queues =  new QVec();
            }
        }
        catch (runtime_error &re) {
            // \todo Handle the error
            active_queues = new QVec();
        }

        return active_queues;
    }

    static QVec& get_opencl_cpu_queues ()
    {
        static QVec* queues = init_queues(backend::opencl,
                                          info::device_type::cpu);
        return *queues;
    }

    static QVec& get_opencl_gpu_queues ()
    {
        static QVec* queues = init_queues(backend::opencl,
                                          info::device_type::gpu);
        return *queues;
    }

    static QVec get_level0_gpu_queues ()
    {
        static QVec* queues = init_queues(backend::level_zero,
                                          info::device_type::gpu);
        return *queues;
    }

    static QVec& get_active_queues ()
    {
        thread_local static QVec *active_queues = init_active_queues();
        return *active_queues;
    }

    static __dppl_give DPPLSyclQueueRef
    getQueue (DPPLSyclBackendType BETy,
              DPPLSyclDeviceType DeviceTy,
              size_t DNum);

    static __dppl_give DPPLSyclQueueRef
    getCurrentQueue ();

    static bool isCurrentQueue (__dppl_keep const DPPLSyclQueueRef QRef);

    static __dppl_give DPPLSyclQueueRef
    setAsDefaultQueue (DPPLSyclBackendType BETy,
                       DPPLSyclDeviceType DeviceTy,
                       size_t DNum);

    static __dppl_give DPPLSyclQueueRef
    pushSyclQueue (DPPLSyclBackendType BETy,
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
    auto &activated_q = get_active_queues();
    if(activated_q.empty()) {
        // \todo handle error
        std::cerr << "No currently active queues.\n";
        return nullptr;
    }
    auto last = activated_q.size() - 1;
    return wrap(new queue(activated_q[last]));
}

/*!
 * Allocates a sycl::queue by copying from the cached {cpu|gpu}_queues vector
 * and returns it to the caller. The caller owns the pointer and is responsible
 * for deallocating it. The helper function DPPLQueue_Delete should
 * be used for that purpose.
 */
__dppl_give DPPLSyclQueueRef
QMgrHelper::getQueue (DPPLSyclBackendType BETy,
                      DPPLSyclDeviceType DeviceTy,
                      size_t DNum)
{
    queue *QRef = nullptr;

    switch (BETy|DeviceTy)
    {
    case DPPLSyclBackendType::DPPL_OPENCL | DPPLSyclDeviceType::DPPL_CPU:
    {
        auto cpuQs = get_opencl_cpu_queues();
        if (DNum >= cpuQs.size()) {
            // \todo handle error
            std::cerr << "OpenCL CPU device " << DNum
                      << " not found on system.\n";
            return nullptr;
        }
        QRef = new queue(cpuQs[DNum]);
        break;
    }
    case DPPLSyclBackendType::DPPL_OPENCL | DPPLSyclDeviceType::DPPL_GPU:
    {
        auto gpuQs = get_opencl_gpu_queues();
        if (DNum >= gpuQs.size()) {
            // \todo handle error
            std::cerr << "OpenCL GPU device " << DNum
                      << " not found on system.\n";
            return nullptr;
        }
        QRef = new queue(gpuQs[DNum]);
        break;
    }
    case DPPLSyclBackendType::DPPL_LEVEL_ZERO | DPPLSyclDeviceType::DPPL_GPU:
    {
        auto l0GpuQs = get_level0_gpu_queues();
        if (DNum >= l0GpuQs.size()) {
            // \todo handle error
            std::cerr << "Level-0 GPU device " << DNum
                      << " not found on system.\n";
            return nullptr;
        }
        QRef = new queue(l0GpuQs[DNum]);
        break;
    }
    default:
        std::cerr << "Unsupported device type.\n";
        return nullptr;
    }

    return wrap(QRef);
}

/*!
 * Compares the context and device of the current queue to the context and
 * device of the queue passed as input. Return true if both queues have the
 * same context and device.
 */
bool QMgrHelper::isCurrentQueue (__dppl_keep const DPPLSyclQueueRef QRef)
{
    auto &activated_q = get_active_queues();
    if(activated_q.empty()) {
        // \todo handle error
        std::cerr << "No currently active queues.\n";
        return false;
    }
    auto last = activated_q.size() - 1;
    auto currQ = activated_q[last];
    return (*unwrap(QRef) == currQ);
}

/*!
 * Changes the first entry into the stack, i.e., the default queue to a new
 * sycl::queue corresponding to the device type and device number.
 */
__dppl_give DPPLSyclQueueRef
QMgrHelper::setAsDefaultQueue (DPPLSyclBackendType BETy,
                               DPPLSyclDeviceType DeviceTy,
                               size_t DNum)
{
    queue *QRef = nullptr;
    auto &activeQ = get_active_queues();
    if(activeQ.empty()) {
        std::cerr << "active queue vector is corrupted.\n";
        return nullptr;
    }

    switch (BETy|DeviceTy)
    {
    case DPPLSyclBackendType::DPPL_OPENCL | DPPLSyclDeviceType::DPPL_CPU:
    {
        auto oclcpu_q = get_opencl_cpu_queues();
        if (DNum >= oclcpu_q.size()) {
            // \todo handle error
            std::cerr << "OpenCL CPU device " << DNum
                      << " not found on system\n.";
            return nullptr;
        }
        activeQ[0] = oclcpu_q[DNum];
        break;
    }
   case DPPLSyclBackendType::DPPL_OPENCL | DPPLSyclDeviceType::DPPL_GPU:
    {
        auto oclgpu_q = get_opencl_gpu_queues();
        if (DNum >= oclgpu_q.size()) {
            // \todo handle error
            std::cerr << "OpenCL GPU device " << DNum
                      << " not found on system\n.";
            return nullptr;
        }
        activeQ[0] = oclgpu_q[DNum];
        break;
    }
    case DPPLSyclBackendType::DPPL_LEVEL_ZERO | DPPLSyclDeviceType::DPPL_GPU:
    {
        auto l0gpu_q = get_level0_gpu_queues();
        if (DNum >= l0gpu_q.size()) {
            // \todo handle error
            std::cerr << "Level-0 GPU device " << DNum
                      << " not found on system\n.";
            return nullptr;
        }
        activeQ[0] = l0gpu_q[DNum];
        break;
    }
    default:
    {
        std::cerr << "Unsupported device type.\n";
        return nullptr;
    }
    }

    QRef = new queue(activeQ[0]);
    return wrap(QRef);
}

/*!
 * Allocates a new sycl::queue by copying from the cached {cpu|gpu}_queues
 * vector. The pointer returned is now owned by the caller and must be properly
 * cleaned up. The helper function DPPLDeleteSyclQueue() can be used is for that
 * purpose.
 */
__dppl_give DPPLSyclQueueRef
QMgrHelper::pushSyclQueue (DPPLSyclBackendType BETy,
                           DPPLSyclDeviceType DeviceTy,
                           size_t DNum)
{
    queue *QRef = nullptr;
    auto &activeQ = get_active_queues();
    if(activeQ.empty()) {
        std::cerr << "Why is there no previous global context?\n";
        return nullptr;
    }

    switch (BETy|DeviceTy)
    {
    case DPPLSyclBackendType::DPPL_OPENCL | DPPLSyclDeviceType::DPPL_CPU:
    {
        if (DNum >= get_opencl_cpu_queues().size()) {
            // \todo handle error
            std::cerr << "OpenCL CPU device " << DNum
                      << " not found on system\n.";
            return nullptr;
        }
        activeQ.emplace_back(get_opencl_cpu_queues()[DNum]);
        QRef = new queue(activeQ[activeQ.size()-1]);
        break;
    }
    case DPPLSyclBackendType::DPPL_OPENCL | DPPLSyclDeviceType::DPPL_GPU:
    {
        if (DNum >= get_opencl_gpu_queues().size()) {
            // \todo handle error
            std::cerr << "OpenCL GPU device " << DNum
                      << " not found on system\n.";
            return nullptr;
        }
        activeQ.emplace_back(get_opencl_gpu_queues()[DNum]);
        QRef = new queue(activeQ[get_active_queues().size()-1]);
        break;
    }
    case DPPLSyclBackendType::DPPL_LEVEL_ZERO | DPPLSyclDeviceType::DPPL_GPU:
    {
        if (DNum >= get_level0_gpu_queues().size()) {
            // \todo handle error
            std::cerr << "Level-0 GPU device " << DNum
                      << " not found on system\n.";
            return nullptr;
        }
        activeQ.emplace_back(get_level0_gpu_queues()[DNum]);
        QRef = new queue(activeQ[get_active_queues().size()-1]);
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
size_t DPPLQueueMgr_GetNumQueues (DPPLSyclBackendType BETy,
                                  DPPLSyclDeviceType DeviceTy)
{
    switch (BETy|DeviceTy)
    {
    case DPPLSyclBackendType::DPPL_OPENCL | DPPLSyclDeviceType::DPPL_CPU:
    {
        return QMgrHelper::get_opencl_cpu_queues().size();
    }
    case DPPLSyclBackendType::DPPL_OPENCL | DPPLSyclDeviceType::DPPL_GPU:
    {
        return QMgrHelper::get_opencl_gpu_queues().size();
    }
    case DPPLSyclBackendType::DPPL_LEVEL_ZERO | DPPLSyclDeviceType::DPPL_GPU:
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
DPPLSyclQueueRef DPPLQueueMgr_GetQueue (DPPLSyclBackendType BETy,
                                        DPPLSyclDeviceType DeviceTy,
                                        size_t DNum)
{
    return QMgrHelper::getQueue(BETy, DeviceTy, DNum);
}

/*!

* */
bool DPPLQueueMgr_IsCurrentQueue (__dppl_keep const DPPLSyclQueueRef QRef)
{
    return QMgrHelper::isCurrentQueue(QRef);
}
/*!
 * The function sets the global queue, i.e., the sycl::queue object at
 * QMgrHelper::active_queues[0] vector to the sycl::queue corresponding to the
 * specified device type and id. If not queue was found for the backend and
 * device, Null is returned.
 */
__dppl_give DPPLSyclQueueRef
DPPLQueueMgr_SetAsDefaultQueue (DPPLSyclBackendType BETy,
                                DPPLSyclDeviceType DeviceTy,
                                size_t DNum)
{
    return QMgrHelper::setAsDefaultQueue(BETy, DeviceTy, DNum);
}

/*!
 * \see QMgrHelper::pushSyclQueue()
 */
__dppl_give DPPLSyclQueueRef
DPPLQueueMgr_PushQueue (DPPLSyclBackendType BETy,
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
