//===----- dpctl_sycl_queue_interface.cpp - Implements C API for sycl::queue =//
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
/// dpctl_sycl_queue_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_queue_interface.h"
#include "../helper/include/dpctl_async_error_handler.h"
#include "Support/CBindingWrapping.h"
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
        // \todo handle errors
        arg_set = false;
        std::cerr << "Kernel argument could not be created.\n";
        break;
    }
    return arg_set;
}

std::unique_ptr<property_list> create_property_list(int properties)
{
    std::unique_ptr<property_list> propList;
    if (properties & (DPCTL_ENABLE_PROFILING | DPCTL_IN_ORDER)) {
        propList = std::make_unique<property_list>(
            sycl::property::queue::enable_profiling(),
            sycl::property::queue::in_order());
    }
    else if (properties & DPCTL_ENABLE_PROFILING) {
        propList = std::make_unique<property_list>(
            sycl::property::queue::enable_profiling());
    }
    else if (properties & DPCTL_IN_ORDER) {
        propList =
            std::make_unique<property_list>(sycl::property::queue::in_order());
    }

    return propList;
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

} /* end of anonymous namespace */

DPCTL_API
__dpctl_give DPCTLSyclQueueRef
DPCTLQueue_Create(__dpctl_keep const DPCTLSyclContextRef CRef,
                  __dpctl_keep const DPCTLSyclDeviceRef DRef,
                  error_handler_callback *error_handler,
                  int properties)
{
    DPCTLSyclQueueRef q = nullptr;
    auto dev = unwrap(DRef);
    auto ctx = unwrap(CRef);

    if (!(dev && ctx)) {
        /* \todo handle error */
        return q;
    }
    auto propList = create_property_list(properties);

    if (propList && error_handler) {
        try {
            auto Queue = new queue(
                *ctx, *dev, DPCTL_AsycErrorHandler(error_handler), *propList);
            q = wrap(Queue);
        } catch (std::bad_alloc const &ba) {
            std::cerr << ba.what() << '\n';
        } catch (runtime_error &re) {
            std::cerr << re.what() << '\n';
        }
    }
    else if (properties) {
        try {
            auto Queue = new queue(*ctx, *dev, *propList);
            q = wrap(Queue);
        } catch (std::bad_alloc const &ba) {
            std::cerr << ba.what() << '\n';
        } catch (runtime_error &re) {
            std::cerr << re.what() << '\n';
        }
    }
    else if (error_handler) {
        try {
            auto Queue =
                new queue(*ctx, *dev, DPCTL_AsycErrorHandler(error_handler));
            q = wrap(Queue);
        } catch (std::bad_alloc const &ba) {
            std::cerr << ba.what() << '\n';
        } catch (runtime_error &re) {
            std::cerr << re.what() << '\n';
        }
    }
    else {
        try {
            auto Queue = new queue(*ctx, *dev);
            q = wrap(Queue);
        } catch (std::bad_alloc const &ba) {
            std::cerr << ba.what() << '\n';
        } catch (runtime_error &re) {
            std::cerr << re.what() << '\n';
        }
    }

    return q;
}

__dpctl_give DPCTLSyclQueueRef
DPCTLQueue_CreateForDevice(__dpctl_keep const DPCTLSyclDeviceRef dRef,
                           error_handler_callback *handler,
                           int properties)
{
    DPCTLSyclQueueRef qRef = nullptr;
    auto Device = unwrap(dRef);

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
    // a new queue to retrun to caller. Note that any context for a sub-device
    // is not cached.
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
        } catch (std::bad_alloc &ba) {
            std::cerr << ba.what() << std::endl;
            return nullptr;
        }
    }
    else {
        std::cerr << "Can not copy DPCTLSyclQueueRef as input is a nullptr"
                  << std::endl;
        return nullptr;
    }
}

bool DPCTLQueue_AreEq(__dpctl_keep const DPCTLSyclQueueRef QRef1,
                      __dpctl_keep const DPCTLSyclQueueRef QRef2)
{
    if (!(QRef1 && QRef2))
        // \todo handle error
        return false;
    return (*unwrap(QRef1) == *unwrap(QRef2));
}

DPCTLSyclBackendType DPCTLQueue_GetBackend(__dpctl_keep DPCTLSyclQueueRef QRef)
{
    auto Q = unwrap(QRef);
    try {
        auto C = Q->get_context();
        return DPCTLContext_GetBackend(wrap(&C));
    } catch (runtime_error &re) {
        std::cerr << re.what() << '\n';
        // store error message
        return DPCTL_UNKNOWN_BACKEND;
    }
}

__dpctl_give DPCTLSyclDeviceRef
DPCTLQueue_GetDevice(__dpctl_keep const DPCTLSyclQueueRef QRef)
{
    auto Q = unwrap(QRef);
    auto Device = new device(Q->get_device());
    return wrap(Device);
}

__dpctl_give DPCTLSyclContextRef
DPCTLQueue_GetContext(__dpctl_keep const DPCTLSyclQueueRef QRef)
{
    auto Q = unwrap(QRef);
    auto Context = new context(Q->get_context());
    return wrap(Context);
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
                // \todo handle errors properly
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
                // \todo handle the error
                throw std::runtime_error("Range cannot be greater than three "
                                         "dimensions.");
            }
        });
    } catch (runtime_error &re) {
        // \todo fix error handling
        std::cerr << re.what() << '\n';
        return nullptr;
    } catch (std::runtime_error &sre) {
        std::cerr << sre.what() << '\n';
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
                // \todo handle errors properly
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
                // \todo handle the error
                throw std::runtime_error("Range cannot be greater than three "
                                         "dimensions.");
            }
        });
    } catch (runtime_error &re) {
        // \todo fix error handling
        std::cerr << re.what() << '\n';
        return nullptr;
    } catch (std::runtime_error &sre) {
        std::cerr << sre.what() << '\n';
        return nullptr;
    }

    return wrap(new event(e));
}

void DPCTLQueue_Wait(__dpctl_keep DPCTLSyclQueueRef QRef)
{
    // \todo what happens if the QRef is null or a pointer to a valid sycl queue
    auto SyclQueue = unwrap(QRef);
    SyclQueue->wait();
}

void DPCTLQueue_Memcpy(__dpctl_keep const DPCTLSyclQueueRef QRef,
                       void *Dest,
                       const void *Src,
                       size_t Count)
{
    auto Q = unwrap(QRef);
    auto event = Q->memcpy(Dest, Src, Count);
    event.wait();
}

void DPCTLQueue_Prefetch(__dpctl_keep DPCTLSyclQueueRef QRef,
                         const void *Ptr,
                         size_t Count)
{
    auto Q = unwrap(QRef);
    auto event = Q->prefetch(Ptr, Count);
    event.wait();
}

void DPCTLQueue_MemAdvise(__dpctl_keep DPCTLSyclQueueRef QRef,
                          const void *Ptr,
                          size_t Count,
                          int Advice)
{
    auto Q = unwrap(QRef);
    auto event = Q->mem_advise(Ptr, Count, static_cast<pi_mem_advice>(Advice));
    event.wait();
}
