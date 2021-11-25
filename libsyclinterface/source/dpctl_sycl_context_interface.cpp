//===- dpctl_sycl_context_interface.cpp - Implements C API for sycl::context =//
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
/// This file implements the data types and functions declared in
/// dpctl_sycl_context_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_context_interface.h"
#include "../helper/include/dpctl_error_handlers.h"
#include "Support/CBindingWrapping.h"
#include <CL/sycl.hpp>
#include <vector>

using namespace cl::sycl;

namespace
{
// Create wrappers for C Binding types (see CBindingWrapping.h).
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(context, DPCTLSyclContextRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(device, DPCTLSyclDeviceRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(std::vector<DPCTLSyclDeviceRef>,
                                   DPCTLDeviceVectorRef)
} /* end of anonymous namespace */

__dpctl_give DPCTLSyclContextRef
DPCTLContext_Create(__dpctl_keep const DPCTLSyclDeviceRef DRef,
                    error_handler_callback *handler,
                    int /**/)
{
    DPCTLSyclContextRef CRef = nullptr;
    auto Device = unwrap(DRef);
    if (!Device) {
        error_handler("Cannot create device from DPCTLSyclDeviceRef"
                      "as input is a nullptr.",
                      __FILE__, __func__, __LINE__);
        return nullptr;
    }
    try {
        CRef = wrap(new context(*Device, DPCTL_AsyncErrorHandler(handler)));
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
    auto DeviceRefs = unwrap(DVRef);
    if (!DeviceRefs) {
        error_handler("Cannot create device reference from DPCTLDeviceVectorRef"
                      "as input is a nullptr.",
                      __FILE__, __func__, __LINE__);
        return CRef;
    }
    Devices.reserve(DeviceRefs->size());

    for (auto const &DRef : *DeviceRefs) {
        Devices.emplace_back(*unwrap(DRef));
    }

    try {
        CRef = wrap(new context(Devices, DPCTL_AsyncErrorHandler(handler)));
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
    return (*unwrap(CtxRef1) == *unwrap(CtxRef2));
}

__dpctl_give DPCTLSyclContextRef
DPCTLContext_Copy(__dpctl_keep const DPCTLSyclContextRef CRef)
{
    auto Context = unwrap(CRef);
    if (!Context) {
        error_handler("Cannot copy DPCTLSyclContextRef as input is a nullptr.",
                      __FILE__, __func__, __LINE__);
        return nullptr;
    }
    try {
        auto CopiedContext = new context(*Context);
        return wrap(CopiedContext);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

__dpctl_give DPCTLDeviceVectorRef
DPCTLContext_GetDevices(__dpctl_keep const DPCTLSyclContextRef CRef)
{
    auto Context = unwrap(CRef);
    if (!Context) {
        error_handler("Cannot retrieve devices from DPCTLSyclContextRef as "
                      "input is a nullptr.",
                      __FILE__, __func__, __LINE__);
        return nullptr;
    }
    std::vector<DPCTLSyclDeviceRef> *DevicesVectorPtr = nullptr;
    try {
        DevicesVectorPtr = new std::vector<DPCTLSyclDeviceRef>();
    } catch (std::exception const &e) {
        delete DevicesVectorPtr;
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
    try {
        auto Devices = Context->get_devices();
        DevicesVectorPtr->reserve(Devices.size());
        for (const auto &Dev : Devices) {
            DevicesVectorPtr->emplace_back(wrap(new device(Dev)));
        }
        return wrap(DevicesVectorPtr);
    } catch (std::exception const &e) {
        delete DevicesVectorPtr;
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

size_t DPCTLContext_DeviceCount(__dpctl_keep const DPCTLSyclContextRef CRef)
{
    auto Context = unwrap(CRef);
    if (!Context) {
        error_handler("Cannot retrieve devices from DPCTLSyclContextRef as "
                      "input is a nullptr.",
                      __FILE__, __func__, __LINE__);
        return 0;
    }
    const auto Devices = Context->get_devices();
    return Devices.size();
}

bool DPCTLContext_IsHost(__dpctl_keep const DPCTLSyclContextRef CtxRef)
{
    auto Ctx = unwrap(CtxRef);
    if (Ctx) {
        return Ctx->is_host();
    }
    return false;
}

void DPCTLContext_Delete(__dpctl_take DPCTLSyclContextRef CtxRef)
{
    delete unwrap(CtxRef);
}

DPCTLSyclBackendType
DPCTLContext_GetBackend(__dpctl_keep const DPCTLSyclContextRef CtxRef)
{
    auto BE = unwrap(CtxRef)->get_platform().get_backend();

    switch (BE) {
    case backend::host:
        return DPCTL_HOST;
    case backend::opencl:
        return DPCTL_OPENCL;
    case backend::level_zero:
        return DPCTL_LEVEL_ZERO;
    case backend::cuda:
        return DPCTL_CUDA;
    default:
        return DPCTL_UNKNOWN_BACKEND;
    }
}

size_t DPCTLContext_Hash(__dpctl_keep const DPCTLSyclContextRef CtxRef)
{
    if (CtxRef) {
        auto C = unwrap(CtxRef);
        std::hash<context> hash_fn;
        return hash_fn(*C);
    }
    else {
        error_handler("Argument CtxRef is null.", __FILE__, __func__, __LINE__);
        return 0;
    }
}
