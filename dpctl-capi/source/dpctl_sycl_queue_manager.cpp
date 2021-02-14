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
#include "Support/CBindingWrapping.h"
#include "dpctl_sycl_device_manager.h"
#include <CL/sycl.hpp> /* SYCL headers   */
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

using QueueStack = vector_class<queue>;

QueueStack &getQueueStack()
{
    thread_local static QueueStack activeQueues =
        QueueStack({default_selector()});
    return activeQueues;
}

template <size_t N>
void updateQueueStack(QueueStack &qs, __dpctl_keep const DPCTLSyclQueueRef qRef)
{
    qs.emplace_back(*unwrap(qRef));
}

template <>
void updateQueueStack<0>(QueueStack &qs,
                         __dpctl_keep const DPCTLSyclQueueRef qRef)
{
    qs[0] = *unwrap(qRef);
}

__dpctl_give DPCTLSyclQueueRef
getQueueImpl(__dpctl_take DPCTLSyclContextRef cRef,
             __dpctl_take DPCTLSyclDeviceRef dRef,
             error_handler_callback *handler,
             int properties)
{
    DPCTLSyclQueueRef qRef = nullptr;
    qRef = DPCTLQueue_Create(cRef, dRef, handler, properties);
    DPCTLContext_Delete(cRef);
    DPCTLDevice_Delete(dRef);

    return qRef;
}

__dpctl_give DPCTLSyclQueueRef
getQueue(__dpctl_keep const DPCTLSyclDeviceRef dRef,
         error_handler_callback *handler,
         int properties)
{
    DPCTLSyclQueueRef qRef = nullptr;
    auto &qs = getQueueStack();
    auto Device = unwrap(dRef);

    if (qs.empty()) {
        std::cerr << "Why is there no previous global context?\n";
        return qRef;
    }
    if (!Device) {
        std::cerr << "Cannot create queue from NULL device reference.\n";
        return qRef;
    }
    auto cached = DPCTLDeviceMgr_GetDeviceAndContextPair(dRef);
    if (cached.CRef) {
        qRef = getQueueImpl(cached.CRef, cached.DRef, handler, properties);
    }
    // We only cache contexts for root devices. If the dRef argument points to
    // a sub-device, then the queue manager allocates a new context and creates
    // a new queue to retrun to caller. Note that the context is not cached.
    else {
        try {
            auto CRef = wrap(new context(*Device));
            auto DRef_copy = wrap(new device(*Device));
            qRef = getQueueImpl(CRef, DRef_copy, handler, properties);
        } catch (std::bad_alloc const &ba) {
            std::cerr << ba.what() << std::endl;
        }
    }

    return qRef;
}

} /* end of anonymous namespace */

//----------------------------- Public API -----------------------------------//

/*!
 * Returns inside the number of activated queues not including the global queue
 * (QMgrHelper::active_queues[0]).
 */
size_t DPCTLQueueMgr_GetNumActivatedQueues()
{
    auto &qs = getQueueStack();
    if (qs.empty()) {
        // \todo handle error
        std::cerr << "No active contexts.\n";
        return 0;
    }
    // The first entry of teh QueueStack is always the global queue. The
    // number of activated queues does not include the global count, that is why
    // we return "size() -1".
    return qs.size() - 1;
}

/*!
 * Allocates a new copy of the present top of stack queue, which can be the
 * default queue and returns to caller. The caller owns the pointer and is
 * responsible for deallocating it. The helper function DPCTLQueue_Delete should
 * be used for that purpose.
 */
DPCTLSyclQueueRef DPCTLQueueMgr_GetCurrentQueue()
{
    auto &qs = getQueueStack();
    if (qs.empty()) {
        // \todo handle error
        std::cerr << "No currently active queues.\n";
        return nullptr;
    }
    auto last = qs.size() - 1;
    return wrap(new queue(qs[last]));
}

/*!
 * Allocates a sycl::queue by copying from the cached {cpu|gpu}_queues vector
 * and returns it to the caller. The caller owns the pointer and is responsible
 * for deallocating it. The helper function DPCTLQueue_Delete should
 * be used for that purpose.
 */
DPCTLSyclQueueRef
DPCTLQueueMgr_GetQueue(__dpctl_keep const DPCTLSyclDeviceRef dRef,
                       error_handler_callback *handler,
                       int properties)
{
    return getQueue(dRef, handler, properties);
}

/*!
 * Compares the context and device of the current queue to the context and
 * device of the queue passed as input. Return true if both queues have the
 * same context and device.
 */
bool DPCTLQueueMgr_IsCurrentQueue(__dpctl_keep const DPCTLSyclQueueRef QRef)
{
    auto &qs = getQueueStack();
    if (qs.empty()) {
        // \todo handle error
        std::cerr << "No currently active queues.\n";
        return false;
    }
    auto last = qs.size() - 1;
    auto currQ = qs[last];
    return (*unwrap(QRef) == currQ);
}

/*!
 * The function sets the global queue, i.e., the sycl::queue object at
 * QMgrHelper::active_queues[0] vector to the sycl::queue corresponding to the
 * specified device type and id. If not queue was found for the backend and
 * device, Null is returned.
 */
__dpctl_give DPCTLSyclQueueRef
DPCTLQueueMgr_SetGlobalQueue(__dpctl_keep const DPCTLSyclDeviceRef dRef,
                             error_handler_callback *handler,
                             int properties)
{
    auto qRef = getQueue(dRef, handler, properties);
    auto qs = getQueueStack();
    if (qRef)
        updateQueueStack<0>(qs, qRef);

    return qRef;
}

/*!
 * Allocates a new sycl::queue by copying from the cached {cpu|gpu}_queues
 * vector. The pointer returned is now owned by the caller and must be properly
 * cleaned up. The helper function DPCTLDeleteSyclQueue() can be used is for
 * that purpose.
 */
__dpctl_give DPCTLSyclQueueRef
DPCTLQueueMgr_PushQueue(__dpctl_keep const DPCTLSyclDeviceRef dRef,
                        error_handler_callback *handler,
                        int properties)
{
    auto qRef = getQueue(dRef, handler, properties);
    auto &qs = getQueueStack();

    if (qRef) {
        updateQueueStack<1>(qs, qRef);
    }
    else {
        std::cerr << "Failed to push the queue to QueueStack.\n";
    }

    return qRef;
}

/*!
 * If there were any sycl::queue that were activated and added to the stack of
 * activated queues then the top of the stack entry is popped. Note that since
 * the same std::vector is used to keep track of the activated queues and the
 * global queue a popSyclQueue call can never make the stack empty. Even
 * after all activated queues are popped, the global queue is still available as
 * the first element added to the stack.
 */
void DPCTLQueueMgr_PopQueue()
{
    auto &qs = getQueueStack();
    // The first queue in the QueueStack is the global queue should not be
    // removed.
    if (qs.size() <= 1) {
        std::cerr << "No queue to pop.\n";
        return;
    }
    qs.pop_back();
}
