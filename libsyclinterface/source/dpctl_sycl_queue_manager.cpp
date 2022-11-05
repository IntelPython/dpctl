//===-------- dpctl_sycl_queue_manager.cpp - Implements a SYCL queue manager =//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2022 Intel Corporation
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
#include "dpctl_error_handlers.h"
#include "dpctl_sycl_device_manager.h"
#include "dpctl_sycl_type_casters.hpp"
#include <CL/sycl.hpp> /* SYCL headers   */
#include <vector>

using namespace sycl;

/*------------------------------- Private helpers ----------------------------*/

// Anonymous namespace for private helpers
namespace
{

using namespace dpctl::syclinterface;

struct QueueManager
{
    using QueueStack = std::vector<queue>;
    static QueueStack &getQueueStack()
    {
        thread_local static QueueStack *activeQueues = new QueueStack([] {
            QueueStack qs;
            auto DS = dpctl_default_selector();
            try {
                auto DRef = wrap<device>(new device(DS));
                auto CRef = DPCTLDeviceMgr_GetCachedContext(DRef);
                if (CRef) {
                    qs.emplace_back(*unwrap<context>(CRef),
                                    *unwrap<device>(DRef));
                }
                else {
                    error_handler("Fatal Error: No cached context for default "
                                  "device.",
                                  __FILE__, __func__, __LINE__);
                    std::terminate();
                }
                delete unwrap<device>(DRef);
                delete unwrap<context>(CRef);
            } catch (std::exception const &e) {
                error_handler(e, __FILE__, __func__, __LINE__);
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
        error_handler("Error: No global queue found.", __FILE__, __func__,
                      __LINE__);
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
        error_handler("No currently active queues.", __FILE__, __func__,
                      __LINE__);
        return nullptr;
    }
    auto last = qs.size() - 1;
    return wrap<queue>(new queue(qs[last]));
}

// Relies on sycl::queue class' operator= to check for equivalent of queues.
bool DPCTLQueueMgr_IsCurrentQueue(__dpctl_keep const DPCTLSyclQueueRef QRef)
{
    if (!QRef) {
        return false;
    }
    auto &qs = QueueManager::getQueueStack();
    if (qs.empty()) {
        error_handler("No currently active queues.", __FILE__, __func__,
                      __LINE__);
        return false;
    }
    auto last = qs.size() - 1;
    auto currQ = qs[last];
    return (*unwrap<queue>(QRef) == currQ);
}

// The function sets the global queue, i.e., the sycl::queue object at
// getQueueStack()[0] to the passed in sycl::queue.
void DPCTLQueueMgr_SetGlobalQueue(__dpctl_keep const DPCTLSyclQueueRef qRef)
{
    auto &qs = QueueManager::getQueueStack();
    if (qRef) {
        qs[0] = *unwrap<queue>(qRef);
    }
    else {
        error_handler("Error: Failed to set the global queue.", __FILE__,
                      __func__, __LINE__);
        std::terminate();
    }
}

// Push the passed in queue to the QueueStack
void DPCTLQueueMgr_PushQueue(__dpctl_keep const DPCTLSyclQueueRef qRef)
{
    auto &qs = QueueManager::getQueueStack();
    if (qRef) {
        qs.emplace_back(*unwrap<queue>(qRef));
    }
    else {
        error_handler("Error: Failed to set the current queue.", __FILE__,
                      __func__, __LINE__);
        std::terminate();
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
        error_handler("No queue to pop.", __FILE__, __func__, __LINE__);
        return;
    }
    qs.pop_back();
}

size_t DPCTLQueueMgr_GetQueueStackSize()
{
    auto &qs = QueueManager::getQueueStack();
    if (qs.empty()) {
        error_handler("Error: No global queue found.", __FILE__, __func__,
                      __LINE__);
        return -1;
    }
    // The first entry of the QueueStack is always the global queue. If there
    // are any more queues in the QueueStack, that indicates that the global
    // queue is not the current queue.
    return (qs.size() - 1);
}
