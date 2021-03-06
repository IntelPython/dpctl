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

struct QueueManager
{
    using QueueStack = vector_class<queue>;
    static QueueStack &getQueueStack()
    {
        thread_local static QueueStack *activeQueues = new QueueStack([] {
            QueueStack qs;
            auto DS = default_selector();
            try {
                auto DRef = wrap(new device(DS.select_device()));
                auto cached = DPCTLDeviceMgr_GetDeviceAndContextPair(DRef);
                if (cached.CRef) {
                    qs.emplace_back(*unwrap(cached.CRef), *unwrap(cached.DRef));
                }
                else {
                    std::cerr << "Fatal Error: No cached context for default "
                                 "device.\n";
                    std::terminate();
                }
                delete unwrap(DRef);
                delete unwrap(cached.DRef);
                delete unwrap(cached.CRef);
            } catch (std::bad_alloc const &ba) {
                std::cerr << ba.what() << '\n';
            }

            return qs;
        }());

        return *activeQueues;
    }
};

} /* end of anonymous namespace */

//----------------------------- Public API -----------------------------------//

// If there are any queues in the QueueStack except the global queue return
// true, else return false.
bool DPCTLQueueMgr_GlobalQueueIsCurrent()
{
    auto &qs = QueueManager::getQueueStack();
    if (qs.empty()) {
        // \todo handle error
        std::cerr << "Error: No global queue found.\n";
        return false;
    }
    // The first entry of the QueueStack is always the global queue. If there
    // are any more queues in the QueueStack, that indicates that the global
    // queue is not the current queue.
    return (qs.size() - 1) ? false : true;
}

/*!
 * Allocates a new copy of the current queue. The caller owns the pointer and is
 * responsible for deallocating it. The helper function DPCTLQueue_Delete should
 * be used for that purpose.
 */
DPCTLSyclQueueRef DPCTLQueueMgr_GetCurrentQueue()
{
    auto &qs = QueueManager::getQueueStack();
    if (qs.empty()) {
        // \todo handle error
        std::cerr << "No currently active queues.\n";
        return nullptr;
    }
    auto last = qs.size() - 1;
    return wrap(new queue(qs[last]));
}

// Relies on sycl::queue class' operator= to check for equivalent of queues.
bool DPCTLQueueMgr_IsCurrentQueue(__dpctl_keep const DPCTLSyclQueueRef QRef)
{
    auto &qs = QueueManager::getQueueStack();
    if (qs.empty()) {
        // \todo handle error
        std::cerr << "No currently active queues.\n";
        return false;
    }
    auto last = qs.size() - 1;
    auto currQ = qs[last];
    return (*unwrap(QRef) == currQ);
}

// The function sets the global queue, i.e., the sycl::queue object at
// getQueueStack()[0] to the passed in sycl::queue.
void DPCTLQueueMgr_SetGlobalQueue(__dpctl_keep const DPCTLSyclQueueRef qRef)
{
    auto &qs = QueueManager::getQueueStack();
    if (qRef) {
        qs[0] = *unwrap(qRef);
    }
    else {
        // TODO: This should be an error and we should not fail silently.
        std::cerr << "Error: Failed to set the global queue.\n";
    }
}

// Push the passed in queue to the QueueStack
void DPCTLQueueMgr_PushQueue(__dpctl_keep const DPCTLSyclQueueRef qRef)
{
    auto &qs = QueueManager::getQueueStack();
    if (qRef) {
        qs.emplace_back(*unwrap(qRef));
    }
    else {
        // TODO: This should be an error and we should not fail silently.
        std::cerr << "Error: Failed to set the current queue.\n";
    }
}

// Pop's a previously pushed queue from the QueueStack. Note that since the
// global queue is always stored at getQueueStack()[0] we check that the size of
// the QueueStack is >=1 before popping.
void DPCTLQueueMgr_PopQueue()
{
    auto &qs = QueueManager::getQueueStack();
    // The first entry in the QueueStack is the global queue, and should not be
    // removed.
    if (qs.size() <= 1) {
        std::cerr << "No queue to pop.\n";
        return;
    }
    qs.pop_back();
}

size_t DPCTLQueueMgr_GetQueueStackSize()
{
    auto &qs = QueueManager::getQueueStack();
    if (qs.empty()) {
        // \todo handle error
        std::cerr << "Error: No global queue found.\n";
        return -1;
    }
    // The first entry of the QueueStack is always the global queue. If there
    // are any more queues in the QueueStack, that indicates that the global
    // queue is not the current queue.
    return (qs.size() - 1);
}
