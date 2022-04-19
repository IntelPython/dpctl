//===- dpctl_sycl_program_interface.cpp - Implements C API for sycl::program =//
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
/// This file implements the functions declared in
/// dpctl_sycl_program_interface.h.
///
//===----------------------------------------------------------------------===//

#ifndef __SYCL_INTERNAL_API
// make sure that sycl::program is defined and implemented
#define __SYCL_INTERNAL_API
#endif

#include "dpctl_sycl_program_interface.h"
#include "Config/dpctl_config.h"
#include "Support/CBindingWrapping.h"
#include "dpctl_error_handlers.h"
#include <CL/cl.h>     /* OpenCL headers     */
#include <CL/sycl.hpp> /* Sycl headers       */
#include <CL/sycl/backend/opencl.hpp>
#include <sstream>

#ifdef DPCTL_ENABLE_L0_PROGRAM_CREATION
#include "dpctl_dynamic_lib_helper.h"
// Note: include ze_api.h before level_zero.hpp. Make sure clang-format does
// not reorder the includes.
// clang-format off
#include "ze_api.h" /* Level Zero headers */
#include "sycl/ext/oneapi/backend/level_zero.hpp"
// clang-format on
#endif

using namespace cl::sycl;

namespace
{
#ifdef DPCTL_ENABLE_L0_PROGRAM_CREATION

#ifdef __linux__
static const char *zeLoaderName = DPCTL_LIBZE_LOADER_FILENAME;
static const int libLoadFlags = RTLD_NOLOAD | RTLD_NOW | RTLD_LOCAL;
#elif defined(_WIN64)
static const char *zeLoaderName = "ze_loader.dll";
static const int libLoadFlags = 0;
#else
#error "Level Zero program compilation is unavailable for this platform"
#endif

typedef ze_result_t (*zeModuleCreateFT)(ze_context_handle_t,
                                        ze_device_handle_t,
                                        const ze_module_desc_t *,
                                        ze_module_handle_t *,
                                        ze_module_build_log_handle_t *);

const char *zeModuleCreateFuncName = "zeModuleCreate";

#endif // #ifdef DPCTL_ENABLE_L0_PROGRAM_CREATION

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(context, DPCTLSyclContextRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(program, DPCTLSyclProgramRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(kernel, DPCTLSyclKernelRef)

__dpctl_give DPCTLSyclProgramRef
createOpenCLInterOpProgram(const context &SyclCtx,
                           __dpctl_keep const void *IL,
                           size_t length,
                           const char *CompileOpts)
{
    cl_int err;
    auto CLCtx = get_native<backend::opencl>(SyclCtx);
    auto CLProgram = clCreateProgramWithIL(CLCtx, IL, length, &err);
    if (err) {
        std::stringstream ss;
        ss << "OpenCL program could not be created from the SPIR-V "
              "binary. OpenCL Error "
           << err << ".";
        error_handler(ss.str(), __FILE__, __func__, __LINE__);
        return nullptr;
    }
    auto SyclDevices = SyclCtx.get_devices();

    // Get a list of CL Devices from the Sycl devices
    auto CLDevices = new cl_device_id[SyclDevices.size()];
    for (auto i = 0ul; i < SyclDevices.size(); ++i)
        CLDevices[i] = get_native<backend::opencl>(SyclDevices[i]);

    // Build the OpenCL interoperability program
    err = clBuildProgram(CLProgram, (cl_uint)(SyclDevices.size()), CLDevices,
                         CompileOpts, nullptr, nullptr);
    // free the CLDevices array
    delete[] CLDevices;

    if (err) {
        std::stringstream ss;
        ss << "OpenCL program could not be built. OpenCL Error " << err << ".";
        error_handler(ss.str(), __FILE__, __func__, __LINE__);
        return nullptr;
    }

    // Create the Sycl program from OpenCL program
    try {
        auto SyclProgram = new program(SyclCtx, CLProgram);
        return wrap(SyclProgram);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

#ifdef DPCTL_ENABLE_L0_PROGRAM_CREATION

zeModuleCreateFT getZeModuleCreateFn()
{
    static dpctl::DynamicLibHelper zeLib(zeLoaderName, libLoadFlags);
    if (!zeLib.opened()) {
        error_handler("The level zero loader dynamic library could not "
                      "be opened.",
                      __FILE__, __func__, __LINE__);
        return nullptr;
    }
    static auto stZeModuleCreateF =
        zeLib.getSymbol<zeModuleCreateFT>(zeModuleCreateFuncName);

    return stZeModuleCreateF;
}

__dpctl_give DPCTLSyclProgramRef
createLevelZeroInterOpProgram(const context &SyclCtx,
                              const void *IL,
                              size_t length,
                              const char *CompileOpts)
{
    auto ZeCtx = get_native<backend::ext_oneapi_level_zero>(SyclCtx);
    auto SyclDevices = SyclCtx.get_devices();
    if (SyclDevices.size() > 1) {
        error_handler("Level zero program can be created for only one device.",
                      __FILE__, __func__, __LINE__);
        return nullptr;
    }

    // Specialization constants are not yet supported.
    // Refer https://bit.ly/33UEDYN for details on specialization constants.
    ze_module_constants_t ZeSpecConstants = {};
    ZeSpecConstants.numConstants = 0;

    // Populate the Level Zero module descriptions
    ze_module_desc_t ZeModuleDesc = {};
    ZeModuleDesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    ZeModuleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    ZeModuleDesc.inputSize = length;
    ZeModuleDesc.pInputModule = (uint8_t *)IL;
    ZeModuleDesc.pBuildFlags = CompileOpts;
    ZeModuleDesc.pConstants = &ZeSpecConstants;

    auto ZeDevice = get_native<backend::ext_oneapi_level_zero>(SyclDevices[0]);
    ze_module_handle_t ZeModule;

    auto stZeModuleCreateF = getZeModuleCreateFn();

    if (!stZeModuleCreateF) {
        error_handler("ZeModuleCreateFn is invalid.", __FILE__, __func__,
                      __LINE__);
        return nullptr;
    }

    auto ret =
        stZeModuleCreateF(ZeCtx, ZeDevice, &ZeModuleDesc, &ZeModule, nullptr);
    if (ret != ZE_RESULT_SUCCESS) {
        error_handler("ZeModule creation failed.", __FILE__, __func__,
                      __LINE__);
        return nullptr;
    }

    // Create the Sycl program from the ZeModule
    try {
        auto ZeProgram =
            new program(sycl::ext::oneapi::level_zero::make_program(
                SyclCtx, reinterpret_cast<uintptr_t>(ZeModule)));
        return wrap(ZeProgram);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}
#endif /* #ifdef DPCTL_ENABLE_L0_PROGRAM_CREATION */

} /* end of anonymous namespace */

__dpctl_give DPCTLSyclProgramRef
DPCTLProgram_CreateFromSpirv(__dpctl_keep const DPCTLSyclContextRef CtxRef,
                             __dpctl_keep const void *IL,
                             size_t length,
                             const char *CompileOpts)
{
    DPCTLSyclProgramRef Pref = nullptr;
    context *SyclCtx = nullptr;
    if (!CtxRef) {
        error_handler("Cannot create program from SPIR-V as the supplied SYCL "
                      "context is NULL.",
                      __FILE__, __func__, __LINE__);
        return Pref;
    }
    SyclCtx = unwrap(CtxRef);
    // get the backend type
    auto BE = SyclCtx->get_platform().get_backend();
    switch (BE) {
    case backend::opencl:
        Pref = createOpenCLInterOpProgram(*SyclCtx, IL, length, CompileOpts);
        break;
    case backend::ext_oneapi_level_zero:
#ifdef DPCTL_ENABLE_L0_PROGRAM_CREATION
        Pref = createLevelZeroInterOpProgram(*SyclCtx, IL, length, CompileOpts);
#endif
        break;
    default:
        break;
    }
    return Pref;
}

__dpctl_give DPCTLSyclProgramRef
DPCTLProgram_CreateFromOCLSource(__dpctl_keep const DPCTLSyclContextRef Ctx,
                                 __dpctl_keep const char *Source,
                                 __dpctl_keep const char *CompileOpts)
{
    std::string compileOpts;
    context *SyclCtx = nullptr;
    program *SyclProgram = nullptr;

    if (!Ctx) {
        error_handler("Input Ctx is nullptr.", __FILE__, __func__, __LINE__);
        return nullptr;
    }

    if (!Source) {
        error_handler("Input Source is nullptr.", __FILE__, __func__, __LINE__);
        return nullptr;
    }

    SyclCtx = unwrap(Ctx);
    SyclProgram = new program(*SyclCtx);
    std::string source = Source;

    if (CompileOpts) {
        compileOpts = CompileOpts;
    }

    // get the backend type
    auto BE = SyclCtx->get_platform().get_backend();
    switch (BE) {
    case backend::opencl:
        try {
            SyclProgram->build_with_source(source, compileOpts);
            return wrap(SyclProgram);
        } catch (std::exception const &e) {
            delete SyclProgram;
            error_handler(e, __FILE__, __func__, __LINE__);
            return nullptr;
        }
        break;
    case backend::ext_oneapi_level_zero:
        error_handler("CreateFromSource is not supported in Level Zero.",
                      __FILE__, __func__, __LINE__);
        delete SyclProgram;
        return nullptr;
    default:
        error_handler("CreateFromSource is not supported in unknown backend.",
                      __FILE__, __func__, __LINE__);
        delete SyclProgram;
        return nullptr;
    }
}

__dpctl_give DPCTLSyclKernelRef
DPCTLProgram_GetKernel(__dpctl_keep DPCTLSyclProgramRef PRef,
                       __dpctl_keep const char *KernelName)
{
    if (!PRef) {
        error_handler("Input PRef is nullptr", __FILE__, __func__, __LINE__);
        return nullptr;
    }
    auto SyclProgram = unwrap(PRef);
    if (!KernelName) {
        error_handler("Input KernelName is nullptr", __FILE__, __func__,
                      __LINE__);
        return nullptr;
    }
    std::string name = KernelName;
    try {
        auto SyclKernel = new kernel(SyclProgram->get_kernel(name));
        return wrap(SyclKernel);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

bool DPCTLProgram_HasKernel(__dpctl_keep DPCTLSyclProgramRef PRef,
                            __dpctl_keep const char *KernelName)
{
    if (!PRef) {
        error_handler("Input PRef is nullptr", __FILE__, __func__, __LINE__);
        return false;
    }
    if (!KernelName) {
        error_handler("Input KernelName is nullptr", __FILE__, __func__,
                      __LINE__);
        return false;
    }

    auto SyclProgram = unwrap(PRef);
    try {
        return SyclProgram->has_kernel(KernelName);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return false;
    }
}

void DPCTLProgram_Delete(__dpctl_take DPCTLSyclProgramRef PRef)
{
    delete unwrap(PRef);
}
