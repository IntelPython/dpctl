//===--- dppl_sycl_program_interface.cpp - DPPL-SYCL interface --*-- C++ -*-===//
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
/// dppl_sycl_program_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dppl_sycl_program_interface.h"
#include "Support/CBindingWrapping.h"

#include <CL/sycl.hpp> /* Sycl headers */
#include <CL/cl.h>     /* OpenCL headers */

using namespace cl::sycl;

namespace
{
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(context, DPPLSyclContextRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(program, DPPLSyclProgramRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(kernel, DPPLSyclKernelRef)

} /* end of anonymous namespace */

__dppl_give DPPLSyclProgramRef
DPPLProgram_CreateFromOCLSpirv (__dppl_keep const DPPLSyclContextRef CtxRef,
                                __dppl_keep const void *IL,
                                size_t length)
{
    cl_int err;
    context *SyclCtx;
    if(!CtxRef) {
        // \todo handle error
        return nullptr;
    }

    SyclCtx = unwrap(CtxRef);
    auto CLCtx   = SyclCtx->get();
    auto CLProgram = clCreateProgramWithIL(CLCtx, IL, length, &err);
    if (err) {
        // \todo: record the error string and any other information.
        std::cerr << "OpenCL program could not be created from the SPIR-V "
                     "binary. OpenCL Error " << err << ".\n";
        return nullptr;
    }
    auto SyclDevices = SyclCtx->get_devices();

    // Get a list of CL Devices from the Sycl devices
    auto CLDevices = new cl_device_id[SyclDevices.size()];
    for (auto i = 0ul; i < SyclDevices.size(); ++i)
        CLDevices[i] = SyclDevices[i].get();

    // Build the OpenCL interoperability program
    err = clBuildProgram(CLProgram, (cl_uint)(SyclDevices.size()), CLDevices,
                         nullptr, nullptr, nullptr);
    // free the CLDevices array
    delete[] CLDevices;

    if (err) {
        // \todo: record the error string and any other information.
        std::cerr << "OpenCL program could not be built. OpenCL Error "
                  << err << ".\n";
        return nullptr;
    }

    // Create the Sycl program from OpenCL program
    try {
        auto SyclProgram = new program(*SyclCtx, CLProgram);
        return wrap(SyclProgram);
    } catch (invalid_object_error) {
        // \todo record error
        return nullptr;
    }
}

__dppl_give DPPLSyclProgramRef
DPPLProgram_CreateFromOCLSource (__dppl_keep const DPPLSyclContextRef Ctx,
                                 __dppl_keep const char *Source,
                                 __dppl_keep const char *CompileOpts)
{
    cl_int err;
    std::string compileOpts;
    context *SyclCtx = nullptr;
    program *SyclProgram = nullptr;

    if(!Ctx) {
        // \todo handle error
        return nullptr;
    }

    if(!Source) {
        // \todo handle error message
        return nullptr;
    }
    std::cout << "CTX address when passed to program creator: " << Ctx << std::endl;

    std::cout << "Ref count in program create : " <<
    unwrap(Ctx)->get_info<info::context::reference_count>() << std::endl;

    queue q{default_selector()};
    SyclCtx = unwrap(Ctx);
    auto syclContext = q.get_context();
    SyclProgram = new program(syclContext);
    std::string source = Source;

    if(CompileOpts) {
        compileOpts = CompileOpts;
    }

    try{
        SyclProgram->build_with_source(source, compileOpts);
        return wrap(SyclProgram);
    } catch (compile_program_error) {
        delete SyclProgram;
        // \todo record error
        return nullptr;
    } catch (feature_not_supported) {
        delete SyclProgram;
        // \todo record error
        return nullptr;
    }
}

__dppl_give DPPLSyclKernelRef
DPPLProgram_GetKernel (__dppl_keep DPPLSyclProgramRef PRef,
                       __dppl_keep const char *KernelName)
{
    if(!PRef) {
        // \todo record error
        return nullptr;
    }
    auto SyclProgram = unwrap(PRef);
    if(!KernelName) {
        // \todo record error
        return nullptr;
    }
    std::string name = KernelName;
    try {
        auto SyclKernel = new kernel(SyclProgram->get_kernel(name));
        return wrap(SyclKernel);
    } catch (invalid_object_error) {
        // \todo record error
        return nullptr;
    }
}

bool
DPPLProgram_HasKernel (__dppl_keep DPPLSyclProgramRef PRef,
                       __dppl_keep const char *KernelName)
{
    if(!PRef) {
        // \todo handle error
        return false;
    }

    auto SyclProgram = unwrap(PRef);
    try {
        return SyclProgram->has_kernel(KernelName);
    } catch (invalid_object_error) {
        return false;
    }
}

void
DPPLProgram_Delete (__dppl_take DPPLSyclProgramRef PRef)
{
    delete unwrap(PRef);
}
