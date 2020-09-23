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
bool set_kernel_arg (handler &cgh, __dppl_keep DPPLKernelArg Arg, size_t idx)
{
    bool arg_set = true;

    switch (Arg.argType)
    {
    case CHAR:
        cgh.set_arg(idx, *(char*)Arg.argVal);
        break;
    case SIGNED_CHAR:
        cgh.set_arg(idx, *(signed char*)Arg.argVal);
        break;
    case UNSIGNED_CHAR:
        cgh.set_arg(idx, *(unsigned char*)Arg.argVal);
        break;
    case SHORT:
        cgh.set_arg(idx, *(short*)Arg.argVal);
        break;
    case INT:
        cgh.set_arg(idx, *(int*)Arg.argVal);
        break;
    case UNSIGNED_INT:
        cgh.set_arg(idx, *(unsigned int*)Arg.argVal);
        break;
    case LONG:
        cgh.set_arg(idx, *(long*)Arg.argVal);
        break;
    case UNSIGNED_LONG:
        cgh.set_arg(idx, *(unsigned long*)Arg.argVal);
        break;
    case LONG_LONG:
        cgh.set_arg(idx, *(long long*)Arg.argVal);
        break;
    case UNSIGNED_LONG_LONG:
        cgh.set_arg(idx, *(unsigned long long*)Arg.argVal);
        break;
    case SIZE_T:
        cgh.set_arg(idx, *(size_t*)Arg.argVal);
        break;
    case FLOAT:
        cgh.set_arg(idx, *(float*)Arg.argVal);
        break;
    case DOUBLE:
        cgh.set_arg(idx, *(double*)Arg.argVal);
        break;
    case LONG_DOUBLE:
        cgh.set_arg(idx, *(long double*)Arg.argVal);
        break;
    case VOID_PTR:
        cgh.set_arg(idx, Arg.argVal);
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
    std::cout << "Context Addr when created in DPPLQueue_GetContext " << Context << '\n';
    std::cout << "Ref count when first created : " <<
    Context->get_info<info::context::reference_count>() << '\n';
    return wrap(Context);
}

__dppl_give DPPLSyclEventRef
DPPLQueue_Submit (__dppl_keep DPPLSyclKernelRef KRef,
                  __dppl_keep DPPLSyclQueueRef QRef,
                  __dppl_keep DPPLKernelArg *Args,
                  size_t NArgs,
                  size_t Range[3],
                  size_t NDims)
{
    auto Kernel = unwrap(KRef);
    auto Queue  = unwrap(QRef);
    event e;

    e = Queue->submit([&](handler& cgh) {
        for (auto i = 0ul; i < 4; ++i) {
            // \todo add support for Sycl buffers
            // \todo handle errors properly
            if(!set_kernel_arg(cgh, Args[i], i))
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
            cgh.parallel_for(range<3>{Range[0], Range[1], Range[2]}, *Kernel);
            break;
        default:
            // \todo handle the error
            std::cerr << "Range cannot be greater than three dimensions.\n";
            exit(1);
        }
    });

    return wrap(new event(e));
}

void
DPPLQueue_Wait (__dppl_keep DPPLSyclQueueRef QRef)
{
    // \todo what happens if the QRef is null or a pointer to a valid sycl queue
    auto SyclQueue = unwrap(QRef);
    SyclQueue->wait();
}
