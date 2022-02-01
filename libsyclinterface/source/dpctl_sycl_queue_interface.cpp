//===----- dpctl_sycl_queue_interface.cpp - Implements C API for sycl::queue =//
//
//                      Data Parallel Control (dpctl)
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
/// dpctl_sycl_queue_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_queue_interface.h"
#include "Support/CBindingWrapping.h"
#include "dpctl_error_handlers.h"
#include "dpctl_sycl_context_interface.h"
#include "dpctl_sycl_device_interface.h"
#include "dpctl_sycl_device_manager.h"
#include <CL/sycl.hpp> /* SYCL headers   */
#include <exception>
#include <stdexcept>

using namespace cl::sycl;

namespace
{
// Create wrappers for C Binding types (see CBindingWrapping.h).
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(context, DPCTLSyclContextRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(device, DPCTLSyclDeviceRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(event, DPCTLSyclEventRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(kernel, DPCTLSyclKernelRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(queue, DPCTLSyclQueueRef)

/*!
 * @brief Set the kernel arg object
 *
 * @param    cgh            My Param doc
 * @param    Arg            My Param doc
 */
bool set_kernel_arg(handler &cgh,
                    size_t idx,
                    __dpctl_keep void *Arg,
                    DPCTLKernelArgType ArgTy)
{
    bool arg_set = true;

    switch (ArgTy) {
    case DPCTL_CHAR:
        cgh.set_arg(idx, *(char *)Arg);
        break;
    case DPCTL_SIGNED_CHAR:
        cgh.set_arg(idx, *(signed char *)Arg);
        break;
    case DPCTL_UNSIGNED_CHAR:
        cgh.set_arg(idx, *(unsigned char *)Arg);
        break;
    case DPCTL_SHORT:
        cgh.set_arg(idx, *(short *)Arg);
        break;
    case DPCTL_INT:
        cgh.set_arg(idx, *(int *)Arg);
        break;
    case DPCTL_UNSIGNED_INT:
        cgh.set_arg(idx, *(unsigned int *)Arg);
        break;
    case DPCTL_UNSIGNED_INT8:
        cgh.set_arg(idx, *(uint8_t *)Arg);
        break;
    case DPCTL_LONG:
        cgh.set_arg(idx, *(long *)Arg);
        break;
    case DPCTL_UNSIGNED_LONG:
        cgh.set_arg(idx, *(unsigned long *)Arg);
        break;
    case DPCTL_LONG_LONG:
        cgh.set_arg(idx, *(long long *)Arg);
        break;
    case DPCTL_UNSIGNED_LONG_LONG:
        cgh.set_arg(idx, *(unsigned long long *)Arg);
        break;
    case DPCTL_SIZE_T:
        cgh.set_arg(idx, *(size_t *)Arg);
        break;
    case DPCTL_FLOAT:
        cgh.set_arg(idx, *(float *)Arg);
        break;
    case DPCTL_DOUBLE:
        cgh.set_arg(idx, *(double *)Arg);
        break;
    case DPCTL_LONG_DOUBLE:
        cgh.set_arg(idx, *(long double *)Arg);
        break;
    case DPCTL_VOID_PTR:
        cgh.set_arg(idx, Arg);
        break;
    default:
        arg_set = false;
        error_handler("Kernel argument could not be created.", __FILE__,
                      __func__, __LINE__);
        break;
    }
    return arg_set;
}

std::unique_ptr<property_list> create_property_list(int properties)
{
    std::unique_ptr<property_list> propList;
    int _prop = properties;
    if (_prop & DPCTL_ENABLE_PROFILING) {
        _prop = _prop ^ DPCTL_ENABLE_PROFILING;
        if (_prop & DPCTL_IN_ORDER) {
            _prop = _prop ^ DPCTL_IN_ORDER;
            propList = std::make_unique<property_list>(
                sycl::property::queue::enable_profiling(),
                sycl::property::queue::in_order());
        }
        else {
            propList = std::make_unique<property_list>(
                sycl::property::queue::enable_profiling());
        }
    }
    else if (_prop & DPCTL_IN_ORDER) {
        _prop = _prop ^ DPCTL_IN_ORDER;
        propList =
            std::make_unique<property_list>(sycl::property::queue::in_order());
    }

    if (_prop) {
        std::stringstream ss;
        ss << "Invalid queue property argument (" << std::hex << properties
           << "), interpreted as (" << (properties ^ _prop) << ").";
        error_handler(ss.str(), __FILE__, __func__, __LINE__);
    }
    return propList;
}

__dpctl_give DPCTLSyclQueueRef
getQueueImpl(__dpctl_keep DPCTLSyclContextRef cRef,
             __dpctl_keep DPCTLSyclDeviceRef dRef,
             error_handler_callback *handler,
             int properties)
{
    DPCTLSyclQueueRef qRef = nullptr;
    qRef = DPCTLQueue_Create(cRef, dRef, handler, properties);
    return qRef;
}

} /* end of anonymous namespace */

DPCTL_API
__dpctl_give DPCTLSyclQueueRef
DPCTLQueue_Create(__dpctl_keep const DPCTLSyclContextRef CRef,
                  __dpctl_keep const DPCTLSyclDeviceRef DRef,
                  error_handler_callback *handler,
                  int properties)
{
    DPCTLSyclQueueRef q = nullptr;
    auto dev = unwrap(DRef);
    auto ctx = unwrap(CRef);

    if (!(dev && ctx)) {
        error_handler("Cannot create queue from DPCTLSyclContextRef and "
                      "DPCTLSyclDeviceRef as input is a nullptr.",
                      __FILE__, __func__, __LINE__);
        return q;
    }
    auto propList = create_property_list(properties);

    if (propList && handler) {
        try {
            auto Queue = new queue(*ctx, *dev, DPCTL_AsyncErrorHandler(handler),
                                   *propList);
            q = wrap(Queue);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    else if (properties) {
        try {
            auto Queue = new queue(*ctx, *dev, *propList);
            q = wrap(Queue);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    else if (handler) {
        try {
            auto Queue =
                new queue(*ctx, *dev, DPCTL_AsyncErrorHandler(handler));
            q = wrap(Queue);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    else {
        try {
            auto Queue = new queue(*ctx, *dev);
            q = wrap(Queue);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }

    return q;
}

__dpctl_give DPCTLSyclQueueRef
DPCTLQueue_CreateForDevice(__dpctl_keep const DPCTLSyclDeviceRef DRef,
                           error_handler_callback *handler,
                           int properties)
{
    DPCTLSyclContextRef CRef = nullptr;
    DPCTLSyclQueueRef QRef = nullptr;
    auto Device = unwrap(DRef);

    if (!Device) {
        error_handler("Cannot create queue from NULL device reference.",
                      __FILE__, __func__, __LINE__);
        return QRef;
    }
    // Check if a cached default context exists for the device.
    CRef = DPCTLDeviceMgr_GetCachedContext(DRef);
    // If a cached default context was found, that context will be used to use
    // create the new queue. When a default cached context was not found, as
    // will be the case for non-root devices, i.e., sub-devices, a new context
    // will be allocated. Note that any newly allocated context is not cached.
    if (!CRef) {
        context *ContextPtr = nullptr;
        try {
            ContextPtr = new context(*Device);
            CRef = wrap(ContextPtr);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
            delete ContextPtr;
            return QRef;
        }
    }
    // At this point we have a valid context and the queue can be allocated.
    QRef = getQueueImpl(CRef, DRef, handler, properties);
    // Free the context
    DPCTLContext_Delete(CRef);
    return QRef;
}

/*!
 * Delete the passed in pointer after verifying it points to a sycl::queue.
 */
void DPCTLQueue_Delete(__dpctl_take DPCTLSyclQueueRef QRef)
{
    delete unwrap(QRef);
}

/*!
 * Make copy of sycl::queue referenced by passed pointer
 */
__dpctl_give DPCTLSyclQueueRef
DPCTLQueue_Copy(__dpctl_keep const DPCTLSyclQueueRef QRef)
{
    auto Queue = unwrap(QRef);
    if (Queue) {
        try {
            auto CopiedQueue = new queue(*Queue);
            return wrap(CopiedQueue);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
            return nullptr;
        }
    }
    else {
        error_handler("Cannot copy DPCTLSyclQueueRef as input is a nullptr",
                      __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

bool DPCTLQueue_AreEq(__dpctl_keep const DPCTLSyclQueueRef QRef1,
                      __dpctl_keep const DPCTLSyclQueueRef QRef2)
{
    if (!(QRef1 && QRef2)) {
        error_handler("DPCTLSyclQueueRefs are nullptr.", __FILE__, __func__,
                      __LINE__);
        return false;
    }
    return (*unwrap(QRef1) == *unwrap(QRef2));
}

DPCTLSyclBackendType DPCTLQueue_GetBackend(__dpctl_keep DPCTLSyclQueueRef QRef)
{
    auto Q = unwrap(QRef);
    if (Q) {
        try {
            auto C = Q->get_context();
            return DPCTLContext_GetBackend(wrap(&C));
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
            return DPCTL_UNKNOWN_BACKEND;
        }
    }
    else
        return DPCTL_UNKNOWN_BACKEND;
}

__dpctl_give DPCTLSyclDeviceRef
DPCTLQueue_GetDevice(__dpctl_keep const DPCTLSyclQueueRef QRef)
{
    DPCTLSyclDeviceRef DRef = nullptr;
    auto Q = unwrap(QRef);
    if (Q) {
        try {
            auto Device = new device(Q->get_device());
            DRef = wrap(Device);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    else {
        error_handler("Could not get the device for this queue.", __FILE__,
                      __func__, __LINE__);
    }
    return DRef;
}

__dpctl_give DPCTLSyclContextRef
DPCTLQueue_GetContext(__dpctl_keep const DPCTLSyclQueueRef QRef)
{
    auto Q = unwrap(QRef);
    DPCTLSyclContextRef CRef = nullptr;
    if (Q)
        CRef = wrap(new context(Q->get_context()));
    else {
        error_handler("Could not get the context for this queue.", __FILE__,
                      __func__, __LINE__);
    }
    return CRef;
}

__dpctl_give DPCTLSyclEventRef
DPCTLQueue_SubmitRange(__dpctl_keep const DPCTLSyclKernelRef KRef,
                       __dpctl_keep const DPCTLSyclQueueRef QRef,
                       __dpctl_keep void **Args,
                       __dpctl_keep const DPCTLKernelArgType *ArgTypes,
                       size_t NArgs,
                       __dpctl_keep const size_t Range[3],
                       size_t NDims,
                       __dpctl_keep const DPCTLSyclEventRef *DepEvents,
                       size_t NDepEvents)
{
    auto Kernel = unwrap(KRef);
    auto Queue = unwrap(QRef);
    event e;

    try {
        e = Queue->submit([&](handler &cgh) {
            // Depend on any event that was specified by the caller.
            if (NDepEvents)
                for (auto i = 0ul; i < NDepEvents; ++i)
                    cgh.depends_on(*unwrap(DepEvents[i]));

            for (auto i = 0ul; i < NArgs; ++i) {
                // \todo add support for Sycl buffers
                if (!set_kernel_arg(cgh, i, Args[i], ArgTypes[i]))
                    exit(1);
            }
            switch (NDims) {
            case 1:
                cgh.parallel_for(range<1>{Range[0]}, *Kernel);
                break;
            case 2:
                cgh.parallel_for(range<2>{Range[0], Range[1]}, *Kernel);
                break;
            case 3:
                cgh.parallel_for(range<3>{Range[0], Range[1], Range[2]},
                                 *Kernel);
                break;
            default:
                throw std::runtime_error("Range cannot be greater than three "
                                         "dimensions.");
            }
        });
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }

    return wrap(new event(e));
}

DPCTLSyclEventRef
DPCTLQueue_SubmitNDRange(__dpctl_keep const DPCTLSyclKernelRef KRef,
                         __dpctl_keep const DPCTLSyclQueueRef QRef,
                         __dpctl_keep void **Args,
                         __dpctl_keep const DPCTLKernelArgType *ArgTypes,
                         size_t NArgs,
                         __dpctl_keep const size_t gRange[3],
                         __dpctl_keep const size_t lRange[3],
                         size_t NDims,
                         __dpctl_keep const DPCTLSyclEventRef *DepEvents,
                         size_t NDepEvents)
{
    auto Kernel = unwrap(KRef);
    auto Queue = unwrap(QRef);
    event e;

    try {
        e = Queue->submit([&](handler &cgh) {
            // Depend on any event that was specified by the caller.
            if (NDepEvents)
                for (auto i = 0ul; i < NDepEvents; ++i)
                    cgh.depends_on(*unwrap(DepEvents[i]));

            for (auto i = 0ul; i < NArgs; ++i) {
                // \todo add support for Sycl buffers
                if (!set_kernel_arg(cgh, i, Args[i], ArgTypes[i]))
                    exit(1);
            }
            switch (NDims) {
            case 1:
                cgh.parallel_for(nd_range<1>{{gRange[0]}, {lRange[0]}},
                                 *Kernel);
                break;
            case 2:
                cgh.parallel_for(
                    nd_range<2>{{gRange[0], gRange[1]}, {lRange[0], lRange[1]}},
                    *Kernel);
                break;
            case 3:
                cgh.parallel_for(nd_range<3>{{gRange[0], gRange[1], gRange[2]},
                                             {lRange[0], lRange[1], lRange[2]}},
                                 *Kernel);
                break;
            default:
                throw std::runtime_error("Range cannot be greater than three "
                                         "dimensions.");
            }
        });
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }

    return wrap(new event(e));
}

void DPCTLQueue_Wait(__dpctl_keep DPCTLSyclQueueRef QRef)
{
    // \todo what happens if the QRef is null or a pointer to a valid sycl
    // queue
    if (QRef) {
        auto SyclQueue = unwrap(QRef);
        if (SyclQueue)
            SyclQueue->wait();
    }
    else {
        error_handler("Argument QRef is NULL.", __FILE__, __func__, __LINE__);
    }
}

DPCTLSyclEventRef DPCTLQueue_Memcpy(__dpctl_keep const DPCTLSyclQueueRef QRef,
                                    void *Dest,
                                    const void *Src,
                                    size_t Count)
{
    auto Q = unwrap(QRef);
    if (Q) {
        sycl::event ev;
        try {
            ev = Q->memcpy(Dest, Src, Count);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
            return nullptr;
        }
        return wrap(new event(ev));
    }
    else {
        error_handler("QRef passed to memcpy was NULL.", __FILE__, __func__,
                      __LINE__);
        return nullptr;
    }
}

DPCTLSyclEventRef DPCTLQueue_Prefetch(__dpctl_keep DPCTLSyclQueueRef QRef,
                                      const void *Ptr,
                                      size_t Count)
{
    auto Q = unwrap(QRef);
    if (Q) {
        if (Ptr) {
            sycl::event ev;
            try {
                ev = Q->prefetch(Ptr, Count);
            } catch (std::exception const &e) {
                error_handler(e, __FILE__, __func__, __LINE__);
                return nullptr;
            }
            return wrap(new event(ev));
        }
        else {
            error_handler("Attempt to prefetch USM-allocation at nullptr.",
                          __FILE__, __func__, __LINE__);
            return nullptr;
        }
    }
    else {
        error_handler("QRef passed to prefetch was NULL.", __FILE__, __func__,
                      __LINE__);
        return nullptr;
    }
}

DPCTLSyclEventRef DPCTLQueue_MemAdvise(__dpctl_keep DPCTLSyclQueueRef QRef,
                                       const void *Ptr,
                                       size_t Count,
                                       int Advice)
{
    auto Q = unwrap(QRef);
    if (Q) {
        sycl::event ev;
        try {
            ev = Q->mem_advise(Ptr, Count, static_cast<pi_mem_advice>(Advice));
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
            return nullptr;
        }
        return wrap(new event(ev));
    }
    else {
        error_handler("QRef passed to prefetch was NULL.", __FILE__, __func__,
                      __LINE__);
        return nullptr;
    }
}

bool DPCTLQueue_IsInOrder(__dpctl_keep const DPCTLSyclQueueRef QRef)
{
    auto Q = unwrap(QRef);
    if (Q) {
        return Q->is_in_order();
    }
    else
        return false;
}

bool DPCTLQueue_HasEnableProfiling(__dpctl_keep const DPCTLSyclQueueRef QRef)
{
    auto Q = unwrap(QRef);
    if (Q) {
        return Q->has_property<sycl::property::queue::enable_profiling>();
    }
    else
        return false;
}

size_t DPCTLQueue_Hash(__dpctl_keep const DPCTLSyclQueueRef QRef)
{
    auto Q = unwrap(QRef);
    if (Q) {
        std::hash<queue> hash_fn;
        return hash_fn(*Q);
    }
    else {
        error_handler("Argument QRef is NULL.", __FILE__, __func__, __LINE__);
        return 0;
    }
}

__dpctl_give DPCTLSyclEventRef DPCTLQueue_SubmitBarrierForEvents(
    __dpctl_keep const DPCTLSyclQueueRef QRef,
    __dpctl_keep const DPCTLSyclEventRef *DepEvents,
    size_t NDepEvents)
{
    auto Q = unwrap(QRef);
    event e;
    if (Q) {
        try {
            e = Q->submit([&](handler &cgh) {
                // Depend on any event that was specified by the caller.
                if (NDepEvents)
                    for (auto i = 0ul; i < NDepEvents; ++i)
                        cgh.depends_on(*unwrap(DepEvents[i]));

                cgh.barrier();
            });
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
            return nullptr;
        }

        return wrap(new event(e));
    }
    else {
        error_handler("Argument QRef is NULL", __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

__dpctl_give DPCTLSyclEventRef
DPCTLQueue_SubmitBarrier(__dpctl_keep const DPCTLSyclQueueRef QRef)
{
    return DPCTLQueue_SubmitBarrierForEvents(QRef, nullptr, 0);
}
