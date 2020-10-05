//===--- dppl_sycl_queue_interface.cpp - DPPL-SYCL interface --*- C++ -*---===//
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
/// dppl_sycl_queue_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dppl_sycl_queue_interface.h"
#include "dppl_sycl_context_interface.h"
#include "Support/CBindingWrapping.h"

#include <CL/sycl.hpp>                /* SYCL headers   */

using namespace cl::sycl;

namespace
{
// Create wrappers for C Binding types (see CBindingWrapping.h).
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(context, DPPLSyclContextRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(device, DPPLSyclDeviceRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(event, DPPLSyclEventRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(kernel, DPPLSyclKernelRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(queue, DPPLSyclQueueRef)

/*!
 * @brief Set the kernel arg object
 *
 * @param    cgh            My Param doc
 * @param    Arg            My Param doc
 */
bool set_kernel_arg (handler &cgh, size_t idx, __dppl_keep void *Arg,
                     DPPLKernelArgType ArgTy)
{
    bool arg_set = true;

    switch (ArgTy)
    {
    case DPPL_CHAR:
        cgh.set_arg(idx, *(char*)Arg);
        break;
    case DPPL_SIGNED_CHAR:
        cgh.set_arg(idx, *(signed char*)Arg);
        break;
    case DPPL_UNSIGNED_CHAR:
        cgh.set_arg(idx, *(unsigned char*)Arg);
        break;
    case DPPL_SHORT:
        cgh.set_arg(idx, *(short*)Arg);
        break;
    case DPPL_INT:
        cgh.set_arg(idx, *(int*)Arg);
        break;
    case DPPL_UNSIGNED_INT:
        cgh.set_arg(idx, *(unsigned int*)Arg);
        break;
    case DPPL_LONG:
        cgh.set_arg(idx, *(long*)Arg);
        break;
    case DPPL_UNSIGNED_LONG:
        cgh.set_arg(idx, *(unsigned long*)Arg);
        break;
    case DPPL_LONG_LONG:
        cgh.set_arg(idx, *(long long*)Arg);
        break;
    case DPPL_UNSIGNED_LONG_LONG:
        cgh.set_arg(idx, *(unsigned long long*)Arg);
        break;
    case DPPL_SIZE_T:
        cgh.set_arg(idx, *(size_t*)Arg);
        break;
    case DPPL_FLOAT:
        cgh.set_arg(idx, *(float*)Arg);
        break;
    case DPPL_DOUBLE:
        cgh.set_arg(idx, *(double*)Arg);
        break;
    case DPPL_LONG_DOUBLE:
        cgh.set_arg(idx, *(long double*)Arg);
        break;
    case DPPL_VOID_PTR:
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

} /* end of anonymous namespace */

/*!
 * Delete the passed in pointer after verifying it points to a sycl::queue.
 */
void DPPLQueue_Delete (__dppl_take DPPLSyclQueueRef QRef)
{
    delete unwrap(QRef);
}


bool DPPLQueue_AreEq (__dppl_keep const DPPLSyclQueueRef QRef1,
                      __dppl_keep const DPPLSyclQueueRef QRef2)
{
    if(!(QRef1 && QRef2))
        // \todo handle error
        return false;
    return (*unwrap(QRef1) == *unwrap(QRef2));
}

DPPLSyclBackendType DPPLQueue_GetBackend (__dppl_keep DPPLSyclQueueRef QRef)
{
    auto Q = unwrap(QRef);
    auto C = Q->get_context();
    return DPPLContext_GetBackend(wrap(&C));
}

__dppl_give DPPLSyclDeviceRef
DPPLQueue_GetDevice (__dppl_keep const DPPLSyclQueueRef QRef)
{
    auto Q = unwrap(QRef);
    auto Device = new device(Q->get_device());
    return wrap(Device);
}

__dppl_give DPPLSyclContextRef
DPPLQueue_GetContext (__dppl_keep const DPPLSyclQueueRef QRef)
{
    auto Q = unwrap(QRef);
    auto Context = new context(Q->get_context());
    return wrap(Context);
}

__dppl_give DPPLSyclEventRef
DPPLQueue_SubmitRange (__dppl_keep const DPPLSyclKernelRef KRef,
                       __dppl_keep const DPPLSyclQueueRef QRef,
                       __dppl_keep void **Args,
                       __dppl_keep const DPPLKernelArgType *ArgTypes,
                       size_t NArgs,
                       __dppl_keep const size_t Range[3],
                       size_t NDims,
                       __dppl_keep const DPPLSyclEventRef *DepEvents,
                       size_t NDepEvents)
{
    auto Kernel = unwrap(KRef);
    auto Queue  = unwrap(QRef);
    event e;

    try {
        e = Queue->submit([&](handler& cgh) {
            // Depend on any event that was specified by the caller.
            if(NDepEvents)
                for(auto i = 0ul; i < NDepEvents; ++i)
                    cgh.depends_on(*unwrap(DepEvents[i]));

            for (auto i = 0ul; i < NArgs; ++i) {
                // \todo add support for Sycl buffers
                // \todo handle errors properly
                if(!set_kernel_arg(cgh, i, Args[i], ArgTypes[i]))
                    exit(1);
            }
            switch(NDims)
            {
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

DPPLSyclEventRef
DPPLQueue_SubmitNDRange(__dppl_keep const DPPLSyclKernelRef KRef,
                        __dppl_keep const DPPLSyclQueueRef QRef,
                        __dppl_keep void **Args,
                        __dppl_keep const DPPLKernelArgType *ArgTypes,
                        size_t NArgs,
                        __dppl_keep const size_t gRange[3],
                        __dppl_keep const size_t lRange[3],
                        size_t NDims,
                        __dppl_keep const DPPLSyclEventRef *DepEvents,
                        size_t NDepEvents)
{
    auto Kernel = unwrap(KRef);
    auto Queue  = unwrap(QRef);
    event e;

    try {
        e = Queue->submit([&](handler& cgh) {
            // Depend on any event that was specified by the caller.
            if(NDepEvents)
                for(auto i = 0ul; i < NDepEvents; ++i)
                    cgh.depends_on(*unwrap(DepEvents[i]));

            for (auto i = 0ul; i < NArgs; ++i) {
                // \todo add support for Sycl buffers
                // \todo handle errors properly
                if(!set_kernel_arg(cgh, i, Args[i], ArgTypes[i]))
                    exit(1);
            }
            switch(NDims)
            {
            case 1:
                cgh.parallel_for(nd_range<1>{{gRange[0]},{lRange[0]}}, *Kernel);
                break;
            case 2:
                cgh.parallel_for(nd_range<2>{{gRange[0], gRange[1]},
                                             {lRange[0], lRange[1]}}, *Kernel);
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

void
DPPLQueue_Wait (__dppl_keep DPPLSyclQueueRef QRef)
{
    // \todo what happens if the QRef is null or a pointer to a valid sycl queue
    auto SyclQueue = unwrap(QRef);
    SyclQueue->wait();
}

void DPPLQueue_Memcpy (__dppl_take const DPPLSyclQueueRef QRef,
                       void *Dest, const void *Src, size_t Count)
{
    auto Q = unwrap(QRef);
    auto event = Q->memcpy(Dest, Src, Count);
    event.wait();
}
