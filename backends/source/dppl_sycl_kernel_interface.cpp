//===--- dppl_sycl_kernel_interface.cpp - DPPL-SYCL interface --*-- C++ -*-===//
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
/// This file implements the functions declared in
/// dppl_sycl_kernel_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dppl_sycl_kernel_interface.h"
#include "Support/CBindingWrapping.h"

#include <CL/sycl.hpp> /* Sycl headers */
#include <CL/cl.h>     /* OpenCL headers */

using namespace cl::sycl;

namespace
{
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(context, DPPLSyclContextRef)
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
        cgh.set_arg(idx, Arg.argVal.char_arg);
        break;
    case SIGNED_CHAR:
        cgh.set_arg(idx, Arg.argVal.schar_arg);
        break;
    case UNSIGNED_CHAR:
        cgh.set_arg(idx, Arg.argVal.uchar_arg);
        break;
    case SHORT:
        cgh.set_arg(idx, Arg.argVal.short_arg);
        break;
    case INT:
        cgh.set_arg(idx, Arg.argVal.int_arg);
        break;
    case UNSIGNED_INT:
        cgh.set_arg(idx, Arg.argVal.uint_arg);
        break;
    case LONG:
        cgh.set_arg(idx, Arg.argVal.long_arg);
        break;
    case UNSIGNED_LONG:
        cgh.set_arg(idx, Arg.argVal.ulong_arg);
        break;
    case LONG_LONG:
        cgh.set_arg(idx, Arg.argVal.longlong_arg);
        break;
    case UNSIGNED_LONG_LONG:
        cgh.set_arg(idx, Arg.argVal.ulonglong_arg);
        break;
    case SIZE_T:
        cgh.set_arg(idx, Arg.argVal.size_t_arg);
        break;
    case FLOAT:
        cgh.set_arg(idx, Arg.argVal.float_arg);
        break;
    case DOUBLE:
        cgh.set_arg(idx, Arg.argVal.double_arg);
        break;
    case LONG_DOUBLE:
        cgh.set_arg(idx, Arg.argVal.longdouble_arg);
        break;
    case CHAR_P:
        cgh.set_arg(idx, Arg.argVal.char_p_arg);
        break;
    case SIGNED_CHAR_P:
        cgh.set_arg(idx, Arg.argVal.schar_p_arg);
        break;
    case UNSIGNED_CHAR_P:
        cgh.set_arg(idx, Arg.argVal.uchar_p_arg);
        break;
    case SHORT_P:
        cgh.set_arg(idx, Arg.argVal.short_p_arg);
        break;
    case INT_P:
        cgh.set_arg(idx, Arg.argVal.int_p_arg);
        break;
    case UNSIGNED_INT_P:
        cgh.set_arg(idx, Arg.argVal.uint_p_arg);
        break;
    case LONG_P:
        cgh.set_arg(idx, Arg.argVal.long_p_arg);
        break;
    case UNSIGNED_LONG_P:
        cgh.set_arg(idx, Arg.argVal.ulong_p_arg);
        break;
    case LONG_LONG_P:
        cgh.set_arg(idx, Arg.argVal.longlong_p_arg);
        break;
    case UNSIGNED_LONG_LONG_P:
        cgh.set_arg(idx, Arg.argVal.ulonglong_p_arg);
        break;
    case SIZE_T_P:
        cgh.set_arg(idx, Arg.argVal.size_t_p_arg);
        break;
    case FLOAT_P:
        cgh.set_arg(idx, Arg.argVal.float_p_arg);
        break;
    case DOUBLE_P:
        cgh.set_arg(idx, Arg.argVal.double_p_arg);
        break;
    case LONG_DOUBLE_P:
        cgh.set_arg(idx, Arg.argVal.longdouble_p_arg);
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

__dppl_give DPPLSyclKernelRef
DPPLKernel_CreateKernelFromSpirv (__dppl_keep const DPPLSyclContextRef Ctx,
                                  __dppl_keep const void *IL,
                                  size_t length,
                                  const char *KernelName)
{
    cl_int err;

    auto SyclCtx = unwrap_context(Ctx);
    auto CLCtx   = SyclCtx->get();
    auto CLProgram = clCreateProgramWithIL(CLCtx, IL, length, &err);
    if (err) {
        // \todo: record the error string and any other information.
        return nullptr;
    }
    auto SyclDevices = SyclCtx->get_devices();

    // Get a list of CL Devices from the Sycl devices
    auto CLDevices = new cl_device_id[SyclDevices.size()];
    for (auto i = 0ul; i < SyclDevices.size(); ++i)
        CLDevices[i] = SyclDevices[i].get();

    // Create the OpenCL interoperability program
    err = clBuildProgram(CLProgram, (cl_uint)(SyclDevices.size()), CLDevices,
                         nullptr, nullptr, nullptr);
    // free the CLDevices array
    delete[] CLDevices;

    if (err) {
        // \todo: record the error string and any other information.
        return nullptr;
    }

    // Create the OpenCL interoperability kernel
    auto CLKernel = clCreateKernel(CLProgram, KernelName, &err);
    if (err) {
        // \todo: record the error string and any other information.
        return nullptr;
    }
    auto SyclKernel = new kernel(CLKernel, *SyclCtx);
    return wrap_kernel(SyclKernel);
}

__dppl_give const char*
DPPLKernel_GetFunctionName (__dppl_keep const DPPLSyclKernelRef Kernel)
{
    auto SyclKernel = unwrap_kernel(Kernel);
    auto kernel_name = SyclKernel->get_info<info::kernel::function_name>();
    if(kernel_name.empty())
        return nullptr;
    auto cstr_name = new char [kernel_name.length()+1];
    std::strcpy (cstr_name, kernel_name.c_str());
    return cstr_name;
}

size_t
DPPLKernel_GetNumArgs (__dppl_keep const DPPLSyclKernelRef Kernel)
{
    auto SyclKernel = unwrap_kernel(Kernel);
    auto num_args = SyclKernel->get_info<info::kernel::num_args>();
    return (size_t)num_args;
}

void
DPPLKernel_DeleteKernelRef (__dppl_take DPPLSyclKernelRef Kernel)
{
    delete unwrap_kernel(Kernel);
}


DPPL_API
DPPLSyclEventRef
DPPLKernel_Submit (__dppl_keep DPPLSyclKernelRef KRef,
                   __dppl_keep DPPLSyclQueueRef QRef,
                   __dppl_keep DPPLKernelArg *Args,
                   size_t NArgs,
                   size_t Range[3],
                   size_t NDims)
{
    auto Kernel = unwrap_kernel(KRef);
    auto Queue  = unwrap_queue(QRef);
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

    return wrap_event(new event(e));
}
