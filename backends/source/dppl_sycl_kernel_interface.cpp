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
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(kernel, DPPLSyclKernelRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(context, DPPLSyclContextRef)
}

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


