//===---- dpctl_sycl_program_interface.cpp - dpctl-C_API  ---*--- C++ --*--===//
//
//               Data Parallel Control Library (dpCtl)
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
/// dpctl_sycl_program_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_program_interface.h"
#include "Support/CBindingWrapping.h"

#include <CL/sycl.hpp> /* Sycl headers */
#include <CL/cl.h>     /* OpenCL headers */

using namespace cl::sycl;

namespace
{
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(context, DPCTLSyclContextRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(program, DPCTLSyclProgramRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(kernel, DPCTLSyclKernelRef)

__dpctl_give DPCTLSyclProgramRef
createOpenCLInterOpProgram (const context &SyclCtx,
                            __dpctl_keep const void *IL,
                            size_t length)
{
    cl_int err;
    auto CLCtx   = SyclCtx.get();
    auto CLProgram = clCreateProgramWithIL(CLCtx, IL, length, &err);
    if (err) {
        // \todo: record the error string and any other information.
        std::cerr << "OpenCL program could not be created from the SPIR-V "
                     "binary. OpenCL Error " << err << ".\n";
        return nullptr;
    }
    auto SyclDevices = SyclCtx.get_devices();

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
        auto SyclProgram = new program(SyclCtx, CLProgram);
        return wrap(SyclProgram);
    } catch (invalid_object_error &e) {
        // \todo record error
        std::cerr << e.what() << '\n';
        return nullptr;
    }
}

} /* end of anonymous namespace */

__dpctl_give DPCTLSyclProgramRef
DPCTLProgram_CreateFromOCLSpirv (__dpctl_keep const DPCTLSyclContextRef CtxRef,
                                 __dpctl_keep const void *IL,
                                 size_t length)
{
    DPCTLSyclProgramRef Pref = nullptr;
    context *SyclCtx = nullptr;
    if(!CtxRef) {
        // \todo handle error
        return Pref;
    }

    SyclCtx = unwrap(CtxRef);
    // get the backend type
    auto BE = SyclCtx->get_platform().get_backend();
    switch (BE)
    {
    case backend::opencl:
        Pref = createOpenCLInterOpProgram(*SyclCtx, IL, length);
        break;
    case backend::level_zero:
        break;
    default:
        break;
    }

    return Pref;
}

__dpctl_give DPCTLSyclProgramRef
DPCTLProgram_CreateFromOCLSource (__dpctl_keep const DPCTLSyclContextRef Ctx,
                                  __dpctl_keep const char *Source,
                                  __dpctl_keep const char *CompileOpts)
{
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

    SyclCtx = unwrap(Ctx);
    SyclProgram = new program(*SyclCtx);
    std::string source = Source;

    if(CompileOpts) {
        compileOpts = CompileOpts;
    }

    // get the backend type
    auto BE = SyclCtx->get_platform().get_backend();
    switch (BE)
    {
    case backend::opencl:
        try {
            SyclProgram->build_with_source(source, compileOpts);
            return wrap(SyclProgram);
        } catch (compile_program_error &e) {
            std::cerr << e.what() << '\n';
            delete SyclProgram;
            // \todo record error
            return nullptr;
        } catch (feature_not_supported &e) {
            std::cerr << e.what() << '\n';
            delete SyclProgram;
            // \todo record error
            return nullptr;
        } catch (runtime_error &e) {
            std::cerr << e.what() << '\n';
            delete SyclProgram;
            // \todo record error
            return nullptr;
        }
        break;
    case backend::level_zero:
        std::cerr << "CreateFromSource is not supported in Level Zero.\n";
        return nullptr;
    default:
        std::cerr << "CreateFromSource is not supported in unknown backend.\n";
        return nullptr;
    }
}

__dpctl_give DPCTLSyclKernelRef
DPCTLProgram_GetKernel (__dpctl_keep DPCTLSyclProgramRef PRef,
                        __dpctl_keep const char *KernelName)
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
    } catch (invalid_object_error &e) {
        // \todo record error
        std::cerr << e.what() << '\n';
        return nullptr;
    }
}

bool
DPCTLProgram_HasKernel (__dpctl_keep DPCTLSyclProgramRef PRef,
                        __dpctl_keep const char *KernelName)
{
    if(!PRef) {
        // \todo handle error
        return false;
    }

    auto SyclProgram = unwrap(PRef);
    try {
        return SyclProgram->has_kernel(KernelName);
    } catch (invalid_object_error &e) {
        std::cerr << e.what() << '\n';
        return false;
    }
}

void
DPCTLProgram_Delete (__dpctl_take DPCTLSyclProgramRef PRef)
{
    delete unwrap(PRef);
}
