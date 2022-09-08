//===- dpctl_sycl_kernel_bundle_interface.cpp - Implements C API for
//    sycl::kernel_bundle<sycl::bundle_state::executable>  ---------------===//
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
/// dpctl_sycl_kernel_bundle_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_kernel_bundle_interface.h"
#include "Config/dpctl_config.h"
#include "Support/CBindingWrapping.h"
#include "dpctl_dynamic_lib_helper.h"
#include "dpctl_error_handlers.h"
#include <CL/cl.h>     /* OpenCL headers     */
#include <CL/sycl.hpp> /* Sycl headers       */
#if __has_include(<sycl/backend/opencl.hpp>)
#include <sycl/backend/opencl.hpp>
#else
#include <CL/sycl/backend/opencl.hpp>
#endif
#include <sstream>

#ifdef DPCTL_ENABLE_L0_PROGRAM_CREATION
// Note: include ze_api.h before level_zero.hpp. Make sure clang-format does
// not reorder the includes.
// clang-format off
#include "ze_api.h" /* Level Zero headers */
#if __has_include(<sycl/backend/level_zero.hpp>)
#include <sycl/backend/level_zero.hpp>
#else
#include <CL/sycl/backend/level_zero.hpp>
#endif
// clang-format on
#endif

using namespace sycl;

namespace
{
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(context, DPCTLSyclContextRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(device, DPCTLSyclDeviceRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(kernel_bundle<bundle_state::executable>,
                                   DPCTLSyclKernelBundleRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(kernel, DPCTLSyclKernelRef)

#ifdef __linux__
static const char *clLoaderName = DPCTL_LIBCL_LOADER_FILENAME;
static const int clLibLoadFlags = RTLD_NOLOAD | RTLD_NOW | RTLD_LOCAL;
#elif defined(_WIN64)
static const char *clLoaderName = "OpenCL.dll";
static const int clLibLoadFlags = 0;
#else
#error "OpenCL program compilation is unavailable for this platform"
#endif

#define CodeStringSuffix(code)                                                 \
    std::string(" (code=") + std::to_string(static_cast<int>(code)) + ")"

#define EnumCaseString(code)                                                   \
    case code:                                                                 \
        return std::string(#code) + CodeStringSuffix(code)

constexpr backend cl_be = backend::opencl;

struct cl_loader
{
public:
    static cl_loader &get()
    {
        static cl_loader _loader;
        return _loader;
    }

    template <typename retTy> retTy getSymbol(const char *name)
    {
        if (!opened) {
            error_handler("The OpenCL loader dynamic library could not "
                          "be opened.",
                          __FILE__, __func__, __LINE__);

            return nullptr;
        }
        return clLib.getSymbol<retTy>(name);
    }

private:
    dpctl::DynamicLibHelper clLib;
    bool opened;
    cl_loader() : clLib(clLoaderName, clLibLoadFlags), opened(clLib.opened()) {}
};

typedef cl_program (*clCreateProgramWithSourceFT)(cl_context,
                                                  cl_uint,
                                                  const char **,
                                                  const size_t *,
                                                  cl_int *);
const char *clCreateProgramWithSource_Name = "clCreateProgramWithSource";
clCreateProgramWithSourceFT get_clCreateProgramWithSource()
{
    static auto st_clCreateProgramWithSourceF =
        cl_loader::get().getSymbol<clCreateProgramWithSourceFT>(
            clCreateProgramWithSource_Name);

    return st_clCreateProgramWithSourceF;
}

typedef cl_program (*clCreateProgramWithILFT)(cl_context,
                                              const void *,
                                              size_t,
                                              cl_int *);
const char *clCreateProgramWithIL_Name = "clCreateProgramWithIL";
clCreateProgramWithILFT get_clCreateProgramWithIL()
{
    static auto st_clCreateProgramWithILF =
        cl_loader::get().getSymbol<clCreateProgramWithILFT>(
            clCreateProgramWithIL_Name);

    return st_clCreateProgramWithILF;
}
typedef cl_int (*clBuildProgramFT)(cl_program,
                                   cl_uint,
                                   const cl_device_id *,
                                   const char *,
                                   void (*)(cl_program, void *),
                                   void *);
const char *clBuildProgram_Name = "clBuildProgram";
clBuildProgramFT get_clBuldProgram()
{
    static auto st_clBuildProgramF =
        cl_loader::get().getSymbol<clBuildProgramFT>(clBuildProgram_Name);

    return st_clBuildProgramF;
}

typedef cl_kernel (*clCreateKernelFT)(cl_program, const char *, cl_int *);
const char *clCreateKernel_Name = "clCreateKernel";
clCreateKernelFT get_clCreateKernel()
{
    static auto st_clCreateKernelF =
        cl_loader::get().getSymbol<clCreateKernelFT>(clCreateKernel_Name);

    return st_clCreateKernelF;
}

std::string _GetErrorCode_ocl_impl(cl_int code)
{
    switch (code) {
        EnumCaseString(CL_BUILD_PROGRAM_FAILURE);
        EnumCaseString(CL_INVALID_CONTEXT);
        EnumCaseString(CL_INVALID_DEVICE);
        EnumCaseString(CL_INVALID_VALUE);
        EnumCaseString(CL_OUT_OF_RESOURCES);
        EnumCaseString(CL_OUT_OF_HOST_MEMORY);
        EnumCaseString(CL_INVALID_OPERATION);
        EnumCaseString(CL_INVALID_BINARY);
    default:
        return "<< ERROR CODE UNRECOGNIZED >>" + CodeStringSuffix(code);
    }
}

DPCTLSyclKernelBundleRef
_CreateKernelBundle_common_ocl_impl(cl_program clProgram,
                                    const context &ctx,
                                    const device &dev,
                                    const char *CompileOpts)
{
    backend_traits<cl_be>::return_type<device> clDevice;
    clDevice = get_native<cl_be>(dev);

    // Last two pointers are notification function pointer and user-data pointer
    // that can be passed to the notification function.
    auto clBuildProgramF = get_clBuldProgram();
    if (clBuildProgramF == nullptr) {
        return nullptr;
    }
    cl_int build_status =
        clBuildProgramF(clProgram, 1, &clDevice, CompileOpts, nullptr, nullptr);

    if (build_status != CL_SUCCESS) {
        error_handler("clBuildProgram failed: " +
                          _GetErrorCode_ocl_impl(build_status),
                      __FILE__, __func__, __LINE__);
        return nullptr;
    }

    kernel_bundle<bundle_state::executable> kb =
        make_kernel_bundle<cl_be, bundle_state::executable>(clProgram, ctx);
    return wrap(new kernel_bundle<bundle_state::executable>(kb));
}

DPCTLSyclKernelBundleRef
_CreateKernelBundleWithOCLSource_ocl_impl(const context &ctx,
                                          const device &dev,
                                          const char *oclSrc,
                                          const char *CompileOpts)
{
    auto clCreateProgramWithSourceF = get_clCreateProgramWithSource();
    if (clCreateProgramWithSourceF == nullptr) {
        return nullptr;
    }

    backend_traits<cl_be>::return_type<context> clContext;
    clContext = get_native<cl_be>(ctx);

    cl_int build_with_source_err_code = CL_SUCCESS;
    cl_program clProgram = clCreateProgramWithSourceF(
        clContext, 1, &oclSrc, nullptr, &build_with_source_err_code);

    if (build_with_source_err_code != CL_SUCCESS) {
        error_handler("clPCreateProgramWithSource failed with " +
                          _GetErrorCode_ocl_impl(build_with_source_err_code),
                      __FILE__, __func__, __LINE__);
        return nullptr;
    }

    return _CreateKernelBundle_common_ocl_impl(clProgram, ctx, dev,
                                               CompileOpts);
}

DPCTLSyclKernelBundleRef
_CreateKernelBundleWithIL_ocl_impl(const context &ctx,
                                   const device &dev,
                                   const void *IL,
                                   size_t il_length,
                                   const char *CompileOpts)
{
    auto clCreateProgramWithILF = get_clCreateProgramWithIL();
    if (clCreateProgramWithILF == nullptr) {
        return nullptr;
    }

    backend_traits<cl_be>::return_type<context> clContext;
    clContext = get_native<cl_be>(ctx);

    cl_int create_err_code = CL_SUCCESS;
    cl_program clProgram =
        clCreateProgramWithILF(clContext, IL, il_length, &create_err_code);

    if (create_err_code != CL_SUCCESS) {
        error_handler("OpenCL program could not be created from the SPIR-V "
                      "binary. OpenCL Error " +
                          _GetErrorCode_ocl_impl(create_err_code),
                      __FILE__, __func__, __LINE__);
        return nullptr;
    }

    return _CreateKernelBundle_common_ocl_impl(clProgram, ctx, dev,
                                               CompileOpts);
}

bool _HasKernel_ocl_impl(const kernel_bundle<bundle_state::executable> &kb,
                         const char *kernel_name)
{
    auto clCreateKernelF = get_clCreateKernel();
    if (clCreateKernelF == nullptr) {
        return false;
    }

    std::vector<cl_program> oclKB = get_native<cl_be>(kb);

    bool found = false;
    for (auto &cl_pr : oclKB) {
        cl_int create_kernel_err_code = CL_SUCCESS;
        [[maybe_unused]] cl_kernel try_kern =
            clCreateKernelF(cl_pr, kernel_name, &create_kernel_err_code);
        if (create_kernel_err_code == CL_SUCCESS) {
            found = true;
            break;
        }
    }
    return found;
}

__dpctl_give DPCTLSyclKernelRef
_GetKernel_ocl_impl(const kernel_bundle<bundle_state::executable> &kb,
                    const char *kernel_name)
{
    auto clCreateKernelF = get_clCreateKernel();
    if (clCreateKernelF == nullptr) {
        return nullptr;
    }

    std::vector<cl_program> oclKB = get_native<cl_be>(kb);

    bool found = false;
    cl_kernel ocl_kernel_from_kb;
    for (auto &cl_pr : oclKB) {
        cl_int create_kernel_err_code = CL_SUCCESS;
        cl_kernel try_kern =
            clCreateKernelF(cl_pr, kernel_name, &create_kernel_err_code);
        if (create_kernel_err_code == CL_SUCCESS) {
            found = true;
            ocl_kernel_from_kb = try_kern;
            break;
        }
    }
    if (found) {
        try {
            context ctx = kb.get_context();

            kernel interop_kernel = make_kernel<cl_be>(ocl_kernel_from_kb, ctx);

            return wrap(new kernel(interop_kernel));
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
            return nullptr;
        }
    }
    else {
        error_handler("Kernel " + std::string(kernel_name) + " not found.",
                      __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

#ifdef DPCTL_ENABLE_L0_PROGRAM_CREATION

#ifdef __linux__
static const char *zeLoaderName = DPCTL_LIBZE_LOADER_FILENAME;
static const int zeLibLoadFlags = RTLD_NOLOAD | RTLD_NOW | RTLD_LOCAL;
#elif defined(_WIN64)
static const char *zeLoaderName = "ze_loader.dll";
static const int zeLibLoadFlags = 0;
#else
#error "Level Zero program compilation is unavailable for this platform"
#endif

constexpr sycl::backend ze_be = sycl::backend::ext_oneapi_level_zero;

struct ze_loader
{
public:
    static ze_loader &get()
    {
        static ze_loader _loader;
        return _loader;
    }

    template <typename retTy> retTy getSymbol(const char *name)
    {
        if (!opened) {
            error_handler("The Level-Zero loader dynamic library could not "
                          "be opened.",
                          __FILE__, __func__, __LINE__);

            return nullptr;
        }
        return zeLib.getSymbol<retTy>(name);
    }

private:
    dpctl::DynamicLibHelper zeLib;
    bool opened;
    ze_loader() : zeLib(zeLoaderName, zeLibLoadFlags), opened(zeLib.opened()) {}
};

typedef ze_result_t (*zeModuleCreateFT)(ze_context_handle_t,
                                        ze_device_handle_t,
                                        const ze_module_desc_t *,
                                        ze_module_handle_t *,
                                        ze_module_build_log_handle_t *);
const char *zeModuleCreate_Name = "zeModuleCreate";
zeModuleCreateFT get_zeModuleCreate()
{
    static auto st_zeModuleCreateF =
        ze_loader::get().getSymbol<zeModuleCreateFT>(zeModuleCreate_Name);

    return st_zeModuleCreateF;
}

typedef ze_result_t (*zeModuleDestroyFT)(ze_module_handle_t);
const char *zeModuleDestroy_Name = "zeModuleDestroy";
zeModuleDestroyFT get_zeModuleDestroy()
{
    static auto st_zeModuleDestroyF =
        ze_loader::get().getSymbol<zeModuleDestroyFT>(zeModuleDestroy_Name);

    return st_zeModuleDestroyF;
}

typedef ze_result_t (*zeKernelCreateFT)(ze_module_handle_t,
                                        const ze_kernel_desc_t *,
                                        ze_kernel_handle_t *);
const char *zeKernelCreate_Name = "zeKernelCreate";
zeKernelCreateFT get_zeKernelCreate()
{
    static auto st_zeKernelCreateF =
        ze_loader::get().getSymbol<zeKernelCreateFT>(zeKernelCreate_Name);

    return st_zeKernelCreateF;
}

std::string _GetErrorCode_ze_impl(ze_result_t code)
{
    switch (code) {
        EnumCaseString(ZE_RESULT_ERROR_UNINITIALIZED);
        EnumCaseString(ZE_RESULT_ERROR_DEVICE_LOST);
        EnumCaseString(ZE_RESULT_ERROR_INVALID_NULL_HANDLE);
        EnumCaseString(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
        EnumCaseString(ZE_RESULT_ERROR_INVALID_ENUMERATION);
        EnumCaseString(ZE_RESULT_ERROR_INVALID_NATIVE_BINARY);
        EnumCaseString(ZE_RESULT_ERROR_INVALID_SIZE);
        EnumCaseString(ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY);
        EnumCaseString(ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY);
        EnumCaseString(ZE_RESULT_ERROR_MODULE_BUILD_FAILURE);
        EnumCaseString(ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED);
    default:
        return "<< UNRECOGNIZED ZE_RESULT_T CODE >> " + CodeStringSuffix(code);
    }
}

__dpctl_give DPCTLSyclKernelBundleRef
_CreateKernelBundleWithIL_ze_impl(const context &SyclCtx,
                                  const device &SyclDev,
                                  const void *IL,
                                  size_t il_length,
                                  const char *CompileOpts)
{
    auto zeModuleCreateFn = get_zeModuleCreate();
    if (zeModuleCreateFn == nullptr) {
        error_handler("ZeModuleCreateFn is invalid.", __FILE__, __func__,
                      __LINE__);
        return nullptr;
    }

    backend_traits<ze_be>::return_type<context> ZeContext;
    ZeContext = get_native<ze_be>(SyclCtx);

    backend_traits<ze_be>::return_type<device> ZeDevice;
    ZeDevice = get_native<ze_be>(SyclDev);

    // Specialization constants are not supported by DPCTL at the moment
    ze_module_constants_t ZeSpecConstants = {};
    ZeSpecConstants.numConstants = 0;

    // Populate the Level Zero module descriptions
    ze_module_desc_t ZeModuleDesc = {};
    ZeModuleDesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    ZeModuleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    ZeModuleDesc.inputSize = il_length;
    ZeModuleDesc.pInputModule = (uint8_t *)IL;
    ZeModuleDesc.pBuildFlags = CompileOpts;
    ZeModuleDesc.pConstants = &ZeSpecConstants;

    ze_module_handle_t ZeModule;

    auto ret_code = zeModuleCreateFn(ZeContext, ZeDevice, &ZeModuleDesc,
                                     &ZeModule, nullptr);
    if (ret_code != ZE_RESULT_SUCCESS) {
        error_handler("Module creation failed " +
                          _GetErrorCode_ze_impl(ret_code),
                      __FILE__, __func__, __LINE__);
        return nullptr;
    }

    try {
        auto kb = make_kernel_bundle<ze_be, bundle_state::executable>(
            {ZeModule, ext::oneapi::level_zero::ownership::keep}, SyclCtx);

        return wrap(new kernel_bundle<bundle_state::executable>(kb));
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        auto zeModuleDestroyFn = get_zeModuleDestroy();
        if (zeModuleDestroyFn) {
            zeModuleDestroyFn(ZeModule);
        }
        return nullptr;
    }
}

__dpctl_give DPCTLSyclKernelRef
_GetKernel_ze_impl(const kernel_bundle<bundle_state::executable> &kb,
                   const char *kernel_name)
{
    auto zeKernelCreateFn = get_zeKernelCreate();
    if (zeKernelCreateFn == nullptr) {
        error_handler("Could not load zeKernelCreate function.", __FILE__,
                      __func__, __LINE__);
        return nullptr;
    }

    auto ZeKernelBundle = sycl::get_native<ze_be>(kb);
    bool found = false;

    // Populate the Level Zero kernel descriptions
    ze_kernel_desc_t ZeKernelDescr = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr,
                                      0, // flags
                                      kernel_name};

    std::unique_ptr<sycl::kernel> syclInteropKern_ptr;
    ze_kernel_handle_t ZeKern;
    for (auto &ZeM : ZeKernelBundle) {
        ze_result_t ze_status = zeKernelCreateFn(ZeM, &ZeKernelDescr, &ZeKern);

        if (ze_status == ZE_RESULT_SUCCESS) {
            found = true;
            auto ctx = kb.get_context();
            auto k = make_kernel<ze_be>(
                {kb, ZeKern, ext::oneapi::level_zero::ownership::keep}, ctx);
            syclInteropKern_ptr = std::unique_ptr<kernel>(new kernel(k));
            break;
        }
        else {
            if (ze_status != ZE_RESULT_ERROR_INVALID_KERNEL_NAME) {
                error_handler("zeKernelCreate failed: " +
                                  _GetErrorCode_ze_impl(ze_status),
                              __FILE__, __func__, __LINE__);
                return nullptr;
            }
        }
    }

    if (found) {
        return wrap(new kernel(*syclInteropKern_ptr));
    }
    else {
        error_handler("Kernel named " + std::string(kernel_name) +
                          " could not be found.",
                      __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

bool _HasKernel_ze_impl(const kernel_bundle<bundle_state::executable> &kb,
                        const char *kernel_name)
{
    auto zeKernelCreateFn = get_zeKernelCreate();
    if (zeKernelCreateFn == nullptr) {
        error_handler("Could not load zeKernelCreate function.", __FILE__,
                      __func__, __LINE__);
        return false;
    }

    auto ZeKernelBundle = sycl::get_native<ze_be>(kb);

    // Populate the Level Zero kernel descriptions
    ze_kernel_desc_t ZeKernelDescr = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr,
                                      0, // flags
                                      kernel_name};

    std::unique_ptr<sycl::kernel> syclInteropKern_ptr;
    ze_kernel_handle_t ZeKern;
    for (auto &ZeM : ZeKernelBundle) {
        ze_result_t ze_status = zeKernelCreateFn(ZeM, &ZeKernelDescr, &ZeKern);

        if (ze_status == ZE_RESULT_SUCCESS) {
            return true;
        }
        else {
            if (ze_status != ZE_RESULT_ERROR_INVALID_KERNEL_NAME) {
                error_handler("zeKernelCreate failed: " +
                                  _GetErrorCode_ze_impl(ze_status),
                              __FILE__, __func__, __LINE__);
                return false;
            }
        }
    }

    return false;
}

#endif /* #ifdef DPCTL_ENABLE_L0_PROGRAM_CREATION */

} /* end of anonymous namespace */

__dpctl_give DPCTLSyclKernelBundleRef
DPCTLKernelBundle_CreateFromSpirv(__dpctl_keep const DPCTLSyclContextRef CtxRef,
                                  __dpctl_keep const DPCTLSyclDeviceRef DevRef,
                                  __dpctl_keep const void *IL,
                                  size_t length,
                                  const char *CompileOpts)
{
    DPCTLSyclKernelBundleRef KBRef = nullptr;
    if (!CtxRef) {
        error_handler("Cannot create program from SPIR-V as the supplied SYCL "
                      "context is NULL.",
                      __FILE__, __func__, __LINE__);
        return KBRef;
    }
    if (!DevRef) {
        error_handler("Cannot create program from SPIR-V as the supplied SYCL "
                      "device is NULL.",
                      __FILE__, __func__, __LINE__);
        return KBRef;
    }
    if ((!IL) || (length == 0)) {
        error_handler("Cannot create program from null SPIR-V buffer.",
                      __FILE__, __func__, __LINE__);
        return KBRef;
    }

    context *SyclCtx = unwrap(CtxRef);
    device *SyclDev = unwrap(DevRef);
    // get the backend type
    auto BE = SyclCtx->get_platform().get_backend();
    switch (BE) {
    case backend::opencl:
        KBRef = _CreateKernelBundleWithIL_ocl_impl(*SyclCtx, *SyclDev, IL,
                                                   length, CompileOpts);
        break;
    case backend::ext_oneapi_level_zero:
#ifdef DPCTL_ENABLE_L0_PROGRAM_CREATION
        KBRef = _CreateKernelBundleWithIL_ze_impl(*SyclCtx, *SyclDev, IL,
                                                  length, CompileOpts);
#endif
        break;
    default:
        error_handler("Backend " + std::to_string(static_cast<int>(BE)) +
                          " is not supported",
                      __FILE__, __func__, __LINE__);
        break;
    }
    return KBRef;
}

__dpctl_give DPCTLSyclKernelBundleRef DPCTLKernelBundle_CreateFromOCLSource(
    __dpctl_keep const DPCTLSyclContextRef Ctx,
    __dpctl_keep const DPCTLSyclDeviceRef Dev,
    __dpctl_keep const char *Source,
    __dpctl_keep const char *CompileOpts)
{
    context *SyclCtx = nullptr;
    device *SyclDev = nullptr;

    if (!Ctx) {
        error_handler("Input Ctx is nullptr.", __FILE__, __func__, __LINE__);
        return nullptr;
    }
    if (!Dev) {
        error_handler("Input Dev is nullptr.", __FILE__, __func__, __LINE__);
        return nullptr;
    }
    if (!Source) {
        error_handler("Input Source is nullptr.", __FILE__, __func__, __LINE__);
        return nullptr;
    }

    SyclCtx = unwrap(Ctx);
    SyclDev = unwrap(Dev);

    // get the backend type
    auto BE = SyclCtx->get_platform().get_backend();
    switch (BE) {
    case backend::opencl:
        try {
            return _CreateKernelBundleWithOCLSource_ocl_impl(
                *SyclCtx, *SyclDev, Source, CompileOpts);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
            return nullptr;
        }
        break;
    case backend::ext_oneapi_level_zero:
        error_handler("CreateFromSource is not supported in Level Zero.",
                      __FILE__, __func__, __LINE__);
        return nullptr;
    default:
        error_handler("CreateFromSource is not supported in unknown backend.",
                      __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

__dpctl_give DPCTLSyclKernelRef
DPCTLKernelBundle_GetKernel(__dpctl_keep DPCTLSyclKernelBundleRef KBRef,
                            __dpctl_keep const char *KernelName)
{
    if (!KBRef) {
        error_handler("Input KBRef is nullptr", __FILE__, __func__, __LINE__);
        return nullptr;
    }
    if (!KernelName) {
        error_handler("Input KernelName is nullptr", __FILE__, __func__,
                      __LINE__);
        return nullptr;
    }
    auto SyclKB = unwrap(KBRef);
    sycl::backend be = SyclKB->get_backend();
    switch (be) {
    case sycl::backend::opencl:
        return _GetKernel_ocl_impl(*SyclKB, KernelName);
    case sycl::backend::ext_oneapi_level_zero:
        return _GetKernel_ze_impl(*SyclKB, KernelName);
    default:
        error_handler("Backend " + std::to_string(static_cast<int>(be)) +
                          " is not supported.",
                      __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

bool DPCTLKernelBundle_HasKernel(__dpctl_keep DPCTLSyclKernelBundleRef KBRef,
                                 __dpctl_keep const char *KernelName)
{
    if (!KBRef) {
        error_handler("Input KBRef is nullptr", __FILE__, __func__, __LINE__);
        return false;
    }
    if (!KernelName) {
        error_handler("Input KernelName is nullptr", __FILE__, __func__,
                      __LINE__);
        return false;
    }

    auto SyclKB = unwrap(KBRef);
    sycl::backend be = SyclKB->get_backend();
    switch (be) {
    case sycl::backend::opencl:
        return _HasKernel_ocl_impl(*SyclKB, KernelName);
    case sycl::backend::ext_oneapi_level_zero:
        return _HasKernel_ze_impl(*SyclKB, KernelName);
    default:
        error_handler("Backend " + std::to_string(static_cast<int>(be)) +
                          " is not supported.",
                      __FILE__, __func__, __LINE__);
        return false;
    }
}

void DPCTLKernelBundle_Delete(__dpctl_take DPCTLSyclKernelBundleRef KBRef)
{
    delete unwrap(KBRef);
}
