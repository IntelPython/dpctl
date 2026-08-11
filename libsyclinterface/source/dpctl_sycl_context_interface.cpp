//===- dpctl_sycl_context_interface.cpp - Implements C API for sycl::context =//
//
//                      Data Parallel Control (dpctl)
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
/// dpctl_sycl_context_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_context_interface.h"
#include "Config/dpctl_config.h"
#include "dpctl_error_handlers.h"
#include "dpctl_sycl_type_casters.hpp"
#include "dpctl_utils_helper.h"
#include <stddef.h>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

using namespace sycl;

namespace
{
static_assert(__SYCL_COMPILER_VERSION >= __SYCL_COMPILER_VERSION_REQUIRED,
              "The compiler does not meet minimum version requirement");

using namespace dpctl::syclinterface;
} // end of anonymous namespace

__dpctl_give DPCTLSyclContextRef
DPCTLContext_Create(__dpctl_keep const DPCTLSyclDeviceRef DRef,
                    error_handler_callback *handler,
                    int /**/)
{
    DPCTLSyclContextRef CRef = nullptr;
    auto Device = unwrap<device>(DRef);
    if (!Device) {
        error_handler("Cannot create device from DPCTLSyclDeviceRef"
                      "as input is a nullptr.",
                      __FILE__, __func__, __LINE__);
        return nullptr;
    }
    try {
        CRef = wrap<context>(
            new context(*Device, DPCTL_AsyncErrorHandler(handler)));
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
    }

    return CRef;
}

__dpctl_give DPCTLSyclContextRef
DPCTLContext_CreateFromDevices(__dpctl_keep const DPCTLDeviceVectorRef DVRef,
                               error_handler_callback *handler,
                               int /**/)
{
    DPCTLSyclContextRef CRef = nullptr;
    std::vector<device> Devices;
    auto DeviceRefs = unwrap<std::vector<DPCTLSyclDeviceRef>>(DVRef);
    if (!DeviceRefs) {
        error_handler("Cannot create device reference from DPCTLDeviceVectorRef"
                      "as input is a nullptr.",
                      __FILE__, __func__, __LINE__);
        return CRef;
    }
    Devices.reserve(DeviceRefs->size());

    for (auto const &DRef : *DeviceRefs) {
        Devices.emplace_back(*unwrap<device>(DRef));
    }

    try {
        CRef = wrap<context>(
            new context(std::move(Devices), DPCTL_AsyncErrorHandler(handler)));
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
    }

    return CRef;
}

bool DPCTLContext_AreEq(__dpctl_keep const DPCTLSyclContextRef CtxRef1,
                        __dpctl_keep const DPCTLSyclContextRef CtxRef2)
{
    if (!(CtxRef1 && CtxRef2)) {
        error_handler("DPCTLSyclContextRefs are nullptr.", __FILE__, __func__,
                      __LINE__);
        return false;
    }
    return (*unwrap<context>(CtxRef1) == *unwrap<context>(CtxRef2));
}

__dpctl_give DPCTLSyclContextRef
DPCTLContext_Copy(__dpctl_keep const DPCTLSyclContextRef CRef)
{
    auto Context = unwrap<context>(CRef);
    if (!Context) {
        error_handler("Cannot copy DPCTLSyclContextRef as input is a nullptr.",
                      __FILE__, __func__, __LINE__);
        return nullptr;
    }
    try {
        auto CopiedContext = new context(*Context);
        return wrap<context>(CopiedContext);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

__dpctl_give DPCTLDeviceVectorRef
DPCTLContext_GetDevices(__dpctl_keep const DPCTLSyclContextRef CRef)
{
    auto Context = unwrap<context>(CRef);
    if (!Context) {
        error_handler("Cannot retrieve devices from DPCTLSyclContextRef as "
                      "input is a nullptr.",
                      __FILE__, __func__, __LINE__);
        return nullptr;
    }
    using vecTy = std::vector<DPCTLSyclDeviceRef>;
    vecTy *DevicesVectorPtr = nullptr;
    try {
        DevicesVectorPtr = new vecTy();
    } catch (std::exception const &e) {
        delete DevicesVectorPtr;
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
    try {
        auto Devices = Context->get_devices();
        DevicesVectorPtr->reserve(Devices.size());
        for (const auto &Dev : Devices) {
            DevicesVectorPtr->emplace_back(
                wrap<device>(new device(std::move(Dev))));
        }
        return wrap<vecTy>(DevicesVectorPtr);
    } catch (std::exception const &e) {
        delete DevicesVectorPtr;
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

size_t DPCTLContext_DeviceCount(__dpctl_keep const DPCTLSyclContextRef CRef)
{
    auto Context = unwrap<context>(CRef);
    if (!Context) {
        error_handler("Cannot retrieve devices from DPCTLSyclContextRef as "
                      "input is a nullptr.",
                      __FILE__, __func__, __LINE__);
        return 0;
    }
    const auto Devices = Context->get_devices();
    return Devices.size();
}

void DPCTLContext_Delete(__dpctl_take DPCTLSyclContextRef CtxRef)
{
    delete unwrap<context>(CtxRef);
}

DPCTLSyclBackendType
DPCTLContext_GetBackend(__dpctl_keep const DPCTLSyclContextRef CtxRef)
{
    if (!CtxRef) {
        return DPCTL_UNKNOWN_BACKEND;
    }

    auto BE = unwrap<context>(CtxRef)->get_platform().get_backend();

    switch (BE) {
    case backend::opencl:
        return DPCTL_OPENCL;
    case backend::ext_oneapi_level_zero:
        return DPCTL_LEVEL_ZERO;
    case backend::ext_oneapi_cuda:
        return DPCTL_CUDA;
    case backend::ext_oneapi_hip:
        return DPCTL_HIP;
    default:
        return DPCTL_UNKNOWN_BACKEND;
    }
}

size_t DPCTLContext_Hash(__dpctl_keep const DPCTLSyclContextRef CtxRef)
{
    if (CtxRef) {
        auto C = unwrap<context>(CtxRef);
        std::hash<context> hash_fn;
        return hash_fn(*C);
    }
    else {
        error_handler("Argument CtxRef is null.", __FILE__, __func__, __LINE__);
        return 0;
    }
}

__dpctl_give DPCTLSyclPlatformRef
DPCTLContext_GetPlatform(__dpctl_keep const DPCTLSyclContextRef CtxRef)
{
    DPCTLSyclPlatformRef PRef = nullptr;
    auto C = unwrap<context>(CtxRef);
    if (C) {
        try {
            PRef = wrap<platform>(
                new platform(C->get_info<info::context::platform>()));
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    return PRef;
}

namespace
{

template <typename InfoDescT, typename ConvertFn>
int *get_context_info_enum_array(__dpctl_keep const DPCTLSyclContextRef CtxRef,
                                 size_t *res_len,
                                 ConvertFn convert)
{
    int *arr = nullptr;
    *res_len = 0;
    auto C = unwrap<context>(CtxRef);
    if (C) {
        try {
            auto values = C->get_info<InfoDescT>();
            *res_len = values.size();
            if (*res_len > 0) {
                arr = new int[*res_len];
                for (size_t i = 0; i < *res_len; ++i) {
                    arr[i] = convert(values[i]);
                }
            }
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
            delete[] arr;
            arr = nullptr;
            *res_len = 0;
        }
    }
    return arr;
}

} // end of anonymous namespace

__dpctl_give int *DPCTLContext_GetAtomicMemoryOrderCapabilities(
    __dpctl_keep const DPCTLSyclContextRef CtxRef,
    size_t *res_len)
{
    return get_context_info_enum_array<
        info::context::atomic_memory_order_capabilities>(
        CtxRef, res_len, DPCTL_SyclMemoryOrderToDPCTLType);
}

__dpctl_give int *DPCTLContext_GetAtomicFenceOrderCapabilities(
    __dpctl_keep const DPCTLSyclContextRef CtxRef,
    size_t *res_len)
{
    return get_context_info_enum_array<
        info::context::atomic_fence_order_capabilities>(
        CtxRef, res_len, DPCTL_SyclMemoryOrderToDPCTLType);
}

__dpctl_give int *DPCTLContext_GetAtomicMemoryScopeCapabilities(
    __dpctl_keep const DPCTLSyclContextRef CtxRef,
    size_t *res_len)
{
    return get_context_info_enum_array<
        info::context::atomic_memory_scope_capabilities>(
        CtxRef, res_len, DPCTL_SyclMemoryScopeToDPCTLType);
}

__dpctl_give int *DPCTLContext_GetAtomicFenceScopeCapabilities(
    __dpctl_keep const DPCTLSyclContextRef CtxRef,
    size_t *res_len)
{
    return get_context_info_enum_array<
        info::context::atomic_fence_scope_capabilities>(
        CtxRef, res_len, DPCTL_SyclMemoryScopeToDPCTLType);
}
