//===-------- dpctl_sycl_queue_manager.cpp - Implements a SYCL queue manager =//
//
//                      Data Parallel Control (dpCtl)
//
// Copyright 2020-2021 Intel Corporation
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
/// dpctl_sycl_queue_manager.h.
///
//===----------------------------------------------------------------------===//
#include "dpctl_sycl_queue_manager.h"
#include "../helper/include/dpctl_utils_helper.h"
#include "Support/CBindingWrapping.h"
#include <CL/sycl.hpp> /* SYCL headers   */
#include <string>
#include <vector>

using namespace cl::sycl;

/*------------------------------- Private helpers ----------------------------*/

// Anonymous namespace for private helpers
namespace
{

// Create wrappers for C Binding types (see CBindingWrapping.h).
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(queue, DPCTLSyclQueueRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(device, DPCTLSyclDeviceRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(context, DPCTLSyclContextRef)

/*
 * Get the number of devices of given type for provided platform.
 */
size_t get_num_devices(const platform &P, backend bty, info::device_type dty)
{
    size_t ndevices = 0;
    if (P.is_host()) {
        if (dty == info::device_type::host)
            ndevices = 1;
    }
    else {
        auto be = P.get_backend();
        if (be == bty) {
            auto Devices = P.get_devices();
            for (auto &Device : Devices) {
                auto devty = Device.get_info<info::device::device_type>();
                if (devty == dty)
                    ++ndevices;
            }
        }
    }

    return ndevices;
}

/*!
 * @brief A helper class to support the DPCTLSyclQueuemanager.
 *
 * The QMgrHelper is needed so that sycl headers are not exposed at the
 * top-level DPCTL API.
 *
 */
class QMgrHelper
{
public:
    using QVec = vector_class<queue>;

    static QVec *init_active_queues()
    {
        QVec *active_queues;
        try {
            queue def_queue{default_selector().select_device()};
            active_queues = new QVec({def_queue});
        } catch (runtime_error &re) {
            // \todo Handle the error
            active_queues = new QVec();
        }

        return active_queues;
    }

    static QVec &get_active_queues()
    {
        thread_local static QVec *active_queues = init_active_queues();
        return *active_queues;
    }

    static __dpctl_give DPCTLSyclQueueRef
    getQueue(__dpctl_keep const DPCTLSyclDeviceRef DRef);

    static __dpctl_give DPCTLSyclQueueRef getCurrentQueue();

    static bool isCurrentQueue(__dpctl_keep const DPCTLSyclQueueRef QRef);

    static __dpctl_give DPCTLSyclQueueRef
    setAsDefaultQueue(__dpctl_keep const DPCTLSyclDeviceRef DRef);

    static __dpctl_give DPCTLSyclQueueRef
    pushSyclQueue(__dpctl_keep const DPCTLSyclDeviceRef DRef);

    static void popSyclQueue();
};

/*!
 * Allocates a new copy of the present top of stack queue, which can be the
 * default queue and returns to caller. The caller owns the pointer and is
 * responsible for deallocating it. The helper function DPCTLQueue_Delete should
 * be used for that purpose.
 */
DPCTLSyclQueueRef QMgrHelper::getCurrentQueue()
{
    auto &activated_q = get_active_queues();
    if (activated_q.empty()) {
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
 * for deallocating it. The helper function DPCTLQueue_Delete should
 * be used for that purpose.
 */
__dpctl_give DPCTLSyclQueueRef
QMgrHelper::getQueue(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    auto Device = unwrap(DRef);
    if (!Device)
        return nullptr;

    // TODO: Implement caching of queues
    try {
        auto QueuePtr = new queue(*Device);
        return wrap(QueuePtr);
    } catch (std::bad_alloc &ba) {
        std::cerr << ba.what() << '\n';
        return nullptr;
    } catch (runtime_error &re) {
        std::cerr << re.what() << '\n';
        return nullptr;
    }
}

/*!
 * Compares the context and device of the current queue to the context and
 * device of the queue passed as input. Return true if both queues have the
 * same context and device.
 */
bool QMgrHelper::isCurrentQueue(__dpctl_keep const DPCTLSyclQueueRef QRef)
{
    auto &activated_q = get_active_queues();
    if (activated_q.empty()) {
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
__dpctl_give DPCTLSyclQueueRef
QMgrHelper::setAsDefaultQueue(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    auto &activeQ = get_active_queues();
    auto Device = unwrap(DRef);
    if (activeQ.empty() || Device) {
        std::cerr << "active queue vector is corrupted.\n";
        return nullptr;
    }

    // TODO: Implement caching of queues
    try {
        auto QueuePtr = new queue(*Device);
        activeQ[0] = *QueuePtr;
        return wrap(QueuePtr);
    } catch (std::bad_alloc &ba) {
        std::cerr << ba.what() << '\n';
        return nullptr;
    } catch (runtime_error &re) {
        std::cerr << re.what() << '\n';
        return nullptr;
    }
}

/*!
 * Allocates a new sycl::queue by copying from the cached {cpu|gpu}_queues
 * vector. The pointer returned is now owned by the caller and must be properly
 * cleaned up. The helper function DPCTLDeleteSyclQueue() can be used is for
 * that purpose.
 */
__dpctl_give DPCTLSyclQueueRef
QMgrHelper::pushSyclQueue(DPCTLSyclDeviceRef DRef)
{
    auto Device = unwrap(DRef);
    auto &activeQ = get_active_queues();
    if (activeQ.empty() || !Device) {
        std::cerr << "Why is there no previous global context?\n";
        return nullptr;
    }

    // TODO: Implement caching of queues
    try {
        auto QueuePtr = new queue(*Device);
        activeQ.emplace_back(*QueuePtr);
        return wrap(QueuePtr);
    } catch (std::bad_alloc &ba) {
        std::cerr << ba.what() << '\n';
        return nullptr;
    } catch (runtime_error &re) {
        std::cerr << re.what() << '\n';
        return nullptr;
    }
}

/*!
 * If there were any sycl::queue that were activated and added to the stack of
 * activated queues then the top of the stack entry is popped. Note that since
 * the same std::vector is used to keep track of the activated queues and the
 * global queue a popSyclQueue call can never make the stack empty. Even
 * after all activated queues are popped, the global queue is still available as
 * the first element added to the stack.
 */
void QMgrHelper::popSyclQueue()
{
    // The first queue which is the "default" queue can not be removed.
    if (get_active_queues().size() <= 1) {
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
size_t DPCTLQueueMgr_GetNumActivatedQueues()
{
    if (QMgrHelper::get_active_queues().empty()) {
        // \todo handle error
        std::cerr << "No active contexts.\n";
        return 0;
    }
    return QMgrHelper::get_active_queues().size() - 1;
}

/*!
 * Returns the number of available devices for a specific backend and device
 * type combination.
 */
size_t DPCTLQueueMgr_GetNumDevices(DPCTLSyclBackendType BETy,
                                   DPCTLSyclDeviceType DeviceTy)
{
    auto Platforms = platform::get_platforms();
    size_t nDevices = 0;

    try {
        auto Backend = DPCTL_DPCTLBackendTypeToSyclBackend(BETy);
        auto DevType = DPCTL_DPCTLDeviceTypeToSyclDeviceType(DeviceTy);
        for (auto &P : Platforms) {
            nDevices = get_num_devices(P, Backend, DevType);
            if (nDevices)
                break;
        }
        return nDevices;
    } catch (runtime_error &re) {
        // \todo log error
        return 0;
    }
}

/*!
 * \see QMgrHelper::getCurrentQueue()
 */
DPCTLSyclQueueRef DPCTLQueueMgr_GetCurrentQueue()
{
    return QMgrHelper::getCurrentQueue();
}

DPCTLSyclQueueRef
DPCTLQueueMgr_GetQueue(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    return QMgrHelper::getQueue(DRef);
}

bool DPCTLQueueMgr_IsCurrentQueue(__dpctl_keep const DPCTLSyclQueueRef QRef)
{
    return QMgrHelper::isCurrentQueue(QRef);
}
/*!
 * The function sets the global queue, i.e., the sycl::queue object at
 * QMgrHelper::active_queues[0] vector to the sycl::queue corresponding to the
 * specified device type and id. If not queue was found for the backend and
 * device, Null is returned.
 */
__dpctl_give DPCTLSyclQueueRef
DPCTLQueueMgr_SetAsDefaultQueue(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    return QMgrHelper::setAsDefaultQueue(DRef);
}

/*!
 * \see QMgrHelper::pushSyclQueue()
 */
__dpctl_give DPCTLSyclQueueRef
DPCTLQueueMgr_PushQueue(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    return QMgrHelper::pushSyclQueue(DRef);
}

/*!
 * \see QMgrHelper::popSyclQueue()
 */
void DPCTLQueueMgr_PopQueue()
{
    QMgrHelper::popSyclQueue();
}

/*!
 * The function constructs a new SYCL queue instance from SYCL conext and
 * SYCL device.
 */
DPCTLSyclQueueRef DPCTLQueueMgr_GetQueueFromContextAndDevice(
    __dpctl_keep DPCTLSyclContextRef CRef,
    __dpctl_keep DPCTLSyclDeviceRef DRef)
{
    auto dev = unwrap(DRef);
    auto ctx = unwrap(CRef);

    return wrap(new queue(*ctx, *dev));
}
