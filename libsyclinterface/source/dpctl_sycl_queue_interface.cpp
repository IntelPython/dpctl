//===----- dpctl_sycl_queue_interface.cpp - Implements C API for sycl::queue =//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2025 Intel Corporation
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
#include "Config/dpctl_config.h"
#include "dpctl_error_handlers.h"
#include "dpctl_sycl_context_interface.h"
#include "dpctl_sycl_device_interface.h"
#include "dpctl_sycl_device_manager.h"
#include "dpctl_sycl_type_casters.hpp"

#include <stddef.h>
#include <stdint.h>

#include <cstdint>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <sycl/sycl.hpp> /* SYCL headers   */
#include <utility>

using namespace sycl;

#define SET_LOCAL_ACCESSOR_ARG(CGH, NDIM, ARGTY, R, IDX)                       \
    do {                                                                       \
        switch ((ARGTY)) {                                                     \
        case DPCTL_INT8_T:                                                     \
        {                                                                      \
            auto la = local_accessor<std::int8_t, NDIM>(R, CGH);               \
            CGH.set_arg(IDX, la);                                              \
            return true;                                                       \
        }                                                                      \
        case DPCTL_UINT8_T:                                                    \
        {                                                                      \
            auto la = local_accessor<std::uint8_t, NDIM>(R, CGH);              \
            CGH.set_arg(IDX, la);                                              \
            return true;                                                       \
        }                                                                      \
        case DPCTL_INT16_T:                                                    \
        {                                                                      \
            auto la = local_accessor<std::int16_t, NDIM>(R, CGH);              \
            CGH.set_arg(IDX, la);                                              \
            return true;                                                       \
        }                                                                      \
        case DPCTL_UINT16_T:                                                   \
        {                                                                      \
            auto la = local_accessor<std::uint16_t, NDIM>(R, CGH);             \
            CGH.set_arg(IDX, la);                                              \
            return true;                                                       \
        }                                                                      \
        case DPCTL_INT32_T:                                                    \
        {                                                                      \
            auto la = local_accessor<std::int32_t, NDIM>(R, CGH);              \
            CGH.set_arg(IDX, la);                                              \
            return true;                                                       \
        }                                                                      \
        case DPCTL_UINT32_T:                                                   \
        {                                                                      \
            auto la = local_accessor<std::uint32_t, NDIM>(R, CGH);             \
            CGH.set_arg(IDX, la);                                              \
            return true;                                                       \
        }                                                                      \
        case DPCTL_INT64_T:                                                    \
        {                                                                      \
            auto la = local_accessor<std::int64_t, NDIM>(R, CGH);              \
            CGH.set_arg(IDX, la);                                              \
            return true;                                                       \
        }                                                                      \
        case DPCTL_UINT64_T:                                                   \
        {                                                                      \
            auto la = local_accessor<std::uint64_t, NDIM>(R, CGH);             \
            CGH.set_arg(IDX, la);                                              \
            return true;                                                       \
        }                                                                      \
        case DPCTL_FLOAT32_T:                                                  \
        {                                                                      \
            auto la = local_accessor<float, NDIM>(R, CGH);                     \
            CGH.set_arg(IDX, la);                                              \
            return true;                                                       \
        }                                                                      \
        case DPCTL_FLOAT64_T:                                                  \
        {                                                                      \
            auto la = local_accessor<double, NDIM>(R, CGH);                    \
            CGH.set_arg(IDX, la);                                              \
            return true;                                                       \
        }                                                                      \
        default:                                                               \
            error_handler("Kernel argument could not be created.", __FILE__,   \
                          __func__, __LINE__, error_level::error);             \
            return false;                                                      \
        }                                                                      \
    } while (0);

namespace
{
static_assert(__SYCL_COMPILER_VERSION >= __SYCL_COMPILER_VERSION_REQUIRED,
              "The compiler does not meet minimum version requirement");

using namespace dpctl::syclinterface;

typedef struct complex
{
    std::uint64_t real;
    std::uint64_t imag;
} complexNumber;

void set_dependent_events(handler &cgh,
                          __dpctl_keep const DPCTLSyclEventRef *DepEvents,
                          size_t NDepEvents)
{
    for (auto i = 0ul; i < NDepEvents; ++i) {
        auto ei = unwrap<event>(DepEvents[i]);
        if (ei)
            cgh.depends_on(*ei);
    }
}

bool set_local_accessor_arg(handler &cgh,
                            size_t idx,
                            const MDLocalAccessor *mdstruct)
{
    switch (mdstruct->ndim) {
    case 1:
    {
        auto r = range<1>(mdstruct->dim0);
        SET_LOCAL_ACCESSOR_ARG(cgh, 1, mdstruct->dpctl_type_id, r, idx);
    }
    case 2:
    {
        auto r = range<2>(mdstruct->dim0, mdstruct->dim1);
        SET_LOCAL_ACCESSOR_ARG(cgh, 2, mdstruct->dpctl_type_id, r, idx);
    }
    case 3:
    {
        auto r = range<3>(mdstruct->dim0, mdstruct->dim1, mdstruct->dim2);
        SET_LOCAL_ACCESSOR_ARG(cgh, 3, mdstruct->dpctl_type_id, r, idx);
    }
    default:
        return false;
    }
}
/*!
 * @brief Set the kernel arg object
 *
 * @param cgh   SYCL command group handler using which a kernel is going to
 *              be submitted.
 * @param idx   The position of the argument in the list of arguments passed
 * to a kernel.
 * @param Arg   A void* representing a kernel argument.
 * @param Argty A typeid specifying the C++ type of the Arg parameter.
 */
bool set_kernel_arg(handler &cgh,
                    size_t idx,
                    __dpctl_keep void *Arg,
                    DPCTLKernelArgType ArgTy)
{
    bool arg_set = true;

    switch (ArgTy) {
    case DPCTL_INT8_T:
        cgh.set_arg(idx, *(std::int8_t *)Arg);
        break;
    case DPCTL_UINT8_T:
        cgh.set_arg(idx, *(std::uint8_t *)Arg);
        break;
    case DPCTL_INT16_T:
        cgh.set_arg(idx, *(std::int16_t *)Arg);
        break;
    case DPCTL_UINT16_T:
        cgh.set_arg(idx, *(std::uint16_t *)Arg);
        break;
    case DPCTL_INT32_T:
        cgh.set_arg(idx, *(std::int32_t *)Arg);
        break;
    case DPCTL_UINT32_T:
        cgh.set_arg(idx, *(std::uint32_t *)Arg);
        break;
    case DPCTL_INT64_T:
        cgh.set_arg(idx, *(std::int64_t *)Arg);
        break;
    case DPCTL_UINT64_T:
        cgh.set_arg(idx, *(std::uint64_t *)Arg);
        break;
    case DPCTL_FLOAT32_T:
        cgh.set_arg(idx, *(float *)Arg);
        break;
    case DPCTL_FLOAT64_T:
        cgh.set_arg(idx, *(double *)Arg);
        break;
    case DPCTL_VOID_PTR:
        cgh.set_arg(idx, Arg);
        break;
    case DPCTL_LOCAL_ACCESSOR:
        arg_set = set_local_accessor_arg(cgh, idx, (MDLocalAccessor *)Arg);
        break;
#if SYCL_EXT_ONEAPI_WORK_GROUP_MEMORY
    case DPCTL_WORK_GROUP_MEMORY:
    {
        size_t num_bytes = reinterpret_cast<std::uintptr_t>(Arg);
        sycl::ext::oneapi::experimental::work_group_memory<char[]> mem{
            num_bytes, cgh};
        cgh.set_arg(idx, mem);
        break;
    }
#endif
    default:
        arg_set = false;
        break;
    }
    return arg_set;
}

void set_kernel_args(handler &cgh,
                     __dpctl_keep void **Args,
                     __dpctl_keep const DPCTLKernelArgType *ArgTypes,
                     size_t NArgs)
{
    for (auto i = 0ul; i < NArgs; ++i) {
        if (!set_kernel_arg(cgh, i, Args[i], ArgTypes[i])) {
            error_handler("Kernel argument could not be created.", __FILE__,
                          __func__, __LINE__);
            throw std::invalid_argument(
                "Kernel argument could not be created.");
        }
    }
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
    else {
        propList = std::make_unique<property_list>();
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
    auto dev = unwrap<device>(DRef);
    auto ctx = unwrap<context>(CRef);

    if (!(dev && ctx)) {
        error_handler("Cannot create queue from DPCTLSyclContextRef and "
                      "DPCTLSyclDeviceRef as input is a nullptr.",
                      __FILE__, __func__, __LINE__);
        return q;
    }
    auto propList = create_property_list(properties);

    if (handler) {
        try {
            auto Queue = new queue(*ctx, *dev, DPCTL_AsyncErrorHandler(handler),
                                   *propList);
            q = wrap<queue>(Queue);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    else {
        try {
            auto Queue = new queue(*ctx, *dev, *propList);
            q = wrap<queue>(Queue);
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
    auto Device = unwrap<device>(DRef);

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
            CRef = wrap<context>(ContextPtr);
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
    delete unwrap<queue>(QRef);
}

/*!
 * Make copy of sycl::queue referenced by passed pointer
 */
__dpctl_give DPCTLSyclQueueRef
DPCTLQueue_Copy(__dpctl_keep const DPCTLSyclQueueRef QRef)
{
    auto Queue = unwrap<queue>(QRef);
    if (Queue) {
        try {
            auto CopiedQueue = new queue(*Queue);
            return wrap<queue>(CopiedQueue);
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
    return (*unwrap<queue>(QRef1) == *unwrap<queue>(QRef2));
}

DPCTLSyclBackendType DPCTLQueue_GetBackend(__dpctl_keep DPCTLSyclQueueRef QRef)
{
    auto Q = unwrap<queue>(QRef);
    if (Q) {
        try {
            auto C = Q->get_context();
            return DPCTLContext_GetBackend(wrap<context>(&C));
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
    auto Q = unwrap<queue>(QRef);
    if (Q) {
        try {
            auto Device = new device(Q->get_device());
            DRef = wrap<device>(Device);
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
    auto Q = unwrap<queue>(QRef);
    DPCTLSyclContextRef CRef = nullptr;
    if (Q)
        CRef = wrap<context>(new context(Q->get_context()));
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
    auto Kernel = unwrap<kernel>(KRef);
    auto Queue = unwrap<queue>(QRef);
    event e;

    try {
        switch (NDims) {
        case 1:
        {
            e = Queue->submit([&](handler &cgh) {
                // Depend on any event that was specified by the caller.
                set_dependent_events(cgh, DepEvents, NDepEvents);
                set_kernel_args(cgh, Args, ArgTypes, NArgs);
                cgh.parallel_for(range<1>{Range[0]}, *Kernel);
            });
            return wrap<event>(new event(std::move(e)));
        }
        case 2:
        {
            e = Queue->submit([&](handler &cgh) {
                // Depend on any event that was specified by the caller.
                set_dependent_events(cgh, DepEvents, NDepEvents);
                set_kernel_args(cgh, Args, ArgTypes, NArgs);
                cgh.parallel_for(range<2>{Range[0], Range[1]}, *Kernel);
            });
            return wrap<event>(new event(std::move(e)));
        }
        case 3:
        {
            e = Queue->submit([&](handler &cgh) {
                // Depend on any event that was specified by the caller.
                set_dependent_events(cgh, DepEvents, NDepEvents);
                set_kernel_args(cgh, Args, ArgTypes, NArgs);
                cgh.parallel_for(range<3>{Range[0], Range[1], Range[2]},
                                 *Kernel);
            });
            return wrap<event>(new event(std::move(e)));
        }
        default:
            error_handler("Range cannot be greater than three "
                          "dimensions.",
                          __FILE__, __func__, __LINE__, error_level::error);
            return nullptr;
        }
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__, error_level::error);
        return nullptr;
    } catch (...) {
        error_handler("Unknown exception encountered", __FILE__, __func__,
                      __LINE__, error_level::error);
        return nullptr;
    }
}

__dpctl_give DPCTLSyclEventRef
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
    auto Kernel = unwrap<kernel>(KRef);
    auto Queue = unwrap<queue>(QRef);
    event e;

    try {
        switch (NDims) {
        case 1:
        {
            e = Queue->submit([&](handler &cgh) {
                // Depend on any event that was specified by the caller.
                set_dependent_events(cgh, DepEvents, NDepEvents);
                set_kernel_args(cgh, Args, ArgTypes, NArgs);
                cgh.parallel_for(nd_range<1>{{gRange[0]}, {lRange[0]}},
                                 *Kernel);
            });
            return wrap<event>(new event(std::move(e)));
        }
        case 2:
        {
            e = Queue->submit([&](handler &cgh) {
                // Depend on any event that was specified by the caller.
                set_dependent_events(cgh, DepEvents, NDepEvents);
                set_kernel_args(cgh, Args, ArgTypes, NArgs);
                cgh.parallel_for(
                    nd_range<2>{{gRange[0], gRange[1]}, {lRange[0], lRange[1]}},
                    *Kernel);
            });
            return wrap<event>(new event(std::move(e)));
        }
        case 3:
        {
            e = Queue->submit([&](handler &cgh) {
                // Depend on any event that was specified by the caller.
                set_dependent_events(cgh, DepEvents, NDepEvents);
                set_kernel_args(cgh, Args, ArgTypes, NArgs);
                cgh.parallel_for(nd_range<3>{{gRange[0], gRange[1], gRange[2]},
                                             {lRange[0], lRange[1], lRange[2]}},
                                 *Kernel);
            });
            return wrap<event>(new event(std::move(e)));
        }
        default:
            error_handler("Range cannot be greater than three "
                          "dimensions.",
                          __FILE__, __func__, __LINE__, error_level::error);
            return nullptr;
        }
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__, error_level::error);
        return nullptr;
    } catch (...) {
        error_handler("Unknown exception encountered", __FILE__, __func__,
                      __LINE__, error_level::error);
        return nullptr;
    }
}

void DPCTLQueue_Wait(__dpctl_keep DPCTLSyclQueueRef QRef)
{
    // \todo what happens if the QRef is null or a pointer to a valid sycl
    // queue
    if (QRef) {
        auto SyclQueue = unwrap<queue>(QRef);
        if (SyclQueue)
            SyclQueue->wait();
    }
    else {
        error_handler("Argument QRef is NULL.", __FILE__, __func__, __LINE__);
    }
}

__dpctl_give DPCTLSyclEventRef
DPCTLQueue_Memcpy(__dpctl_keep const DPCTLSyclQueueRef QRef,
                  void *Dest,
                  const void *Src,
                  size_t Count)
{
    auto Q = unwrap<queue>(QRef);
    if (Q) {
        sycl::event ev;
        try {
            ev = Q->memcpy(Dest, Src, Count);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
            return nullptr;
        }
        return wrap<event>(new event(std::move(ev)));
    }
    else {
        error_handler("QRef passed to memcpy was NULL.", __FILE__, __func__,
                      __LINE__);
        return nullptr;
    }
}

__dpctl_give DPCTLSyclEventRef
DPCTLQueue_MemcpyWithEvents(__dpctl_keep const DPCTLSyclQueueRef QRef,
                            void *Dest,
                            const void *Src,
                            size_t Count,
                            const DPCTLSyclEventRef *DepEvents,
                            size_t DepEventsCount)
{
    event ev;
    auto Q = unwrap<queue>(QRef);
    if (Q) {
        try {
            ev = Q->submit([&](handler &cgh) {
                if (DepEvents)
                    for (size_t i = 0; i < DepEventsCount; ++i) {
                        event *ei = unwrap<event>(DepEvents[i]);
                        if (ei)
                            cgh.depends_on(*ei);
                    }

                cgh.memcpy(Dest, Src, Count);
            });
        } catch (const std::exception &ex) {
            error_handler(ex, __FILE__, __func__, __LINE__);
            return nullptr;
        }
    }
    else {
        error_handler("QRef passed to memcpy was NULL.", __FILE__, __func__,
                      __LINE__);
        return nullptr;
    }

    return wrap<event>(new event(ev));
}

__dpctl_give DPCTLSyclEventRef
DPCTLQueue_Prefetch(__dpctl_keep DPCTLSyclQueueRef QRef,
                    const void *Ptr,
                    size_t Count)
{
    auto Q = unwrap<queue>(QRef);
    if (Q) {
        if (Ptr) {
            sycl::event ev;
            try {
                ev = Q->prefetch(Ptr, Count);
            } catch (std::exception const &e) {
                error_handler(e, __FILE__, __func__, __LINE__);
                return nullptr;
            }
            return wrap<event>(new event(std::move(ev)));
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

__dpctl_give DPCTLSyclEventRef
DPCTLQueue_MemAdvise(__dpctl_keep DPCTLSyclQueueRef QRef,
                     const void *Ptr,
                     size_t Count,
                     int Advice)
{
    auto Q = unwrap<queue>(QRef);
    if (Q) {
        sycl::event ev;
        try {
            ev = Q->mem_advise(Ptr, Count, Advice);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
            return nullptr;
        }
        return wrap<event>(new event(std::move(ev)));
    }
    else {
        error_handler("QRef passed to prefetch was NULL.", __FILE__, __func__,
                      __LINE__);
        return nullptr;
    }
}

bool DPCTLQueue_IsInOrder(__dpctl_keep const DPCTLSyclQueueRef QRef)
{
    auto Q = unwrap<queue>(QRef);
    if (Q) {
        return Q->is_in_order();
    }
    else
        return false;
}

bool DPCTLQueue_HasEnableProfiling(__dpctl_keep const DPCTLSyclQueueRef QRef)
{
    auto Q = unwrap<queue>(QRef);
    if (Q) {
        return Q->has_property<sycl::property::queue::enable_profiling>();
    }
    else
        return false;
}

size_t DPCTLQueue_Hash(__dpctl_keep const DPCTLSyclQueueRef QRef)
{
    auto Q = unwrap<queue>(QRef);
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
    auto Q = unwrap<queue>(QRef);
    event e;
    if (Q) {
        try {
            e = Q->submit([&](handler &cgh) {
                // Depend on any event that was specified by the caller.
                if (NDepEvents)
                    for (auto i = 0ul; i < NDepEvents; ++i)
                        cgh.depends_on(*unwrap<event>(DepEvents[i]));

                cgh.ext_oneapi_barrier();
            });
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
            return nullptr;
        }

        return wrap<event>(new event(std::move(e)));
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

__dpctl_give DPCTLSyclEventRef
DPCTLQueue_Memset(__dpctl_keep const DPCTLSyclQueueRef QRef,
                  void *USMRef,
                  uint8_t Value,
                  size_t Count)
{
    auto Q = unwrap<queue>(QRef);
    if (Q && USMRef) {
        sycl::event ev;
        try {
            ev = Q->memset(USMRef, static_cast<int>(Value), Count);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
            return nullptr;
        }
        return wrap<event>(new event(std::move(ev)));
    }
    else {
        error_handler("QRef or USMRef passed to fill8 were NULL.", __FILE__,
                      __func__, __LINE__);
        return nullptr;
    }
};

__dpctl_give DPCTLSyclEventRef
DPCTLQueue_Fill8(__dpctl_keep const DPCTLSyclQueueRef QRef,
                 void *USMRef,
                 uint8_t Value,
                 size_t Count)
{
    auto Q = unwrap<queue>(QRef);
    if (Q && USMRef) {
        sycl::event ev;
        try {
            ev = Q->fill<uint8_t>(USMRef, Value, Count);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
            return nullptr;
        }
        return wrap<event>(new event(std::move(ev)));
    }
    else {
        error_handler("QRef or USMRef passed to fill8 were NULL.", __FILE__,
                      __func__, __LINE__);
        return nullptr;
    }
}

__dpctl_give DPCTLSyclEventRef
DPCTLQueue_Fill16(__dpctl_keep const DPCTLSyclQueueRef QRef,
                  void *USMRef,
                  uint16_t Value,
                  size_t Count)
{
    auto Q = unwrap<queue>(QRef);
    if (Q && USMRef) {
        sycl::event ev;
        try {
            ev = Q->fill<uint16_t>(USMRef, Value, Count);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
            return nullptr;
        }
        return wrap<event>(new event(std::move(ev)));
    }
    else {
        error_handler("QRef or USMRef passed to fill16 were NULL.", __FILE__,
                      __func__, __LINE__);
        return nullptr;
    }
}

__dpctl_give DPCTLSyclEventRef
DPCTLQueue_Fill32(__dpctl_keep const DPCTLSyclQueueRef QRef,
                  void *USMRef,
                  uint32_t Value,
                  size_t Count)
{
    auto Q = unwrap<queue>(QRef);
    if (Q && USMRef) {
        sycl::event ev;
        try {
            ev = Q->fill<uint32_t>(USMRef, Value, Count);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
            return nullptr;
        }
        return wrap<event>(new event(std::move(ev)));
    }
    else {
        error_handler("QRef or USMRef passed to fill32 were NULL.", __FILE__,
                      __func__, __LINE__);
        return nullptr;
    }
}

__dpctl_give DPCTLSyclEventRef
DPCTLQueue_Fill64(__dpctl_keep const DPCTLSyclQueueRef QRef,
                  void *USMRef,
                  uint64_t Value,
                  size_t Count)
{
    auto Q = unwrap<queue>(QRef);
    if (Q && USMRef) {
        sycl::event ev;
        try {
            ev = Q->fill<uint64_t>(USMRef, Value, Count);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
            return nullptr;
        }
        return wrap<event>(new event(std::move(ev)));
    }
    else {
        error_handler("QRef or USMRef passed to fill64 were NULL.", __FILE__,
                      __func__, __LINE__);
        return nullptr;
    }
}

__dpctl_give DPCTLSyclEventRef
DPCTLQueue_Fill128(__dpctl_keep const DPCTLSyclQueueRef QRef,
                   void *USMRef,
                   uint64_t *Value,
                   size_t Count)
{
    auto Q = unwrap<queue>(QRef);
    if (Q && USMRef) {
        sycl::event ev;
        try {
            complexNumber Val;
            Val.real = Value[0];
            Val.imag = Value[1];
            ev = Q->fill(USMRef, Val, Count);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
            return nullptr;
        }
        return wrap<event>(new event(std::move(ev)));
    }
    else {
        error_handler("QRef or USMRef passed to fill128 were NULL.", __FILE__,
                      __func__, __LINE__);
        return nullptr;
    }
}
