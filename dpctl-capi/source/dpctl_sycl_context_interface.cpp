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
#include "../helper/include/dpctl_async_error_handler.h"
#include "Support/CBindingWrapping.h"
#include <CL/sycl.hpp>

using namespace cl::sycl;

namespace
{
// Create wrappers for C Binding types (see CBindingWrapping.h).
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(context, DPCTLSyclContextRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(device, DPCTLSyclDeviceRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(vector_class<DPCTLSyclDeviceRef>,
                                   DPCTLDeviceVectorRef)
} /* end of anonymous namespace */

__dpctl_give DPCTLSyclContextRef
DPCTLContext_Create(__dpctl_keep const DPCTLSyclDeviceRef DRef,
                    error_handler_callback *error_handler,
                    int /**/)
{
    DPCTLSyclContextRef CRef = nullptr;
    auto Device = unwrap(DRef);
    if (!Device)
        return CRef;
    try {
        CRef =
            wrap(new context(*Device, DPCTL_AsyncErrorHandler(error_handler)));
    } catch (const std::bad_alloc &ba) {
        std::cerr << ba.what() << '\n';
    } catch (const runtime_error &re) {
        std::cerr << re.what() << '\n';
    }

    return CRef;
}

__dpctl_give DPCTLSyclContextRef
DPCTLContext_CreateFromDevices(__dpctl_keep const DPCTLDeviceVectorRef DVRef,
                               error_handler_callback *error_handler,
                               int /**/)
{
    DPCTLSyclContextRef CRef = nullptr;
    vector_class<device> Devices;
    auto DeviceRefs = unwrap(DVRef);
    if (!DeviceRefs)
        return CRef;
    Devices.reserve(DeviceRefs->size());

    for (auto const &DRef : *DeviceRefs) {
        Devices.emplace_back(*unwrap(DRef));
    }

    try {
        CRef =
            wrap(new context(Devices, DPCTL_AsyncErrorHandler(error_handler)));
    } catch (const std::bad_alloc &ba) {
        std::cerr << ba.what() << '\n';
    } catch (const runtime_error &re) {
        std::cerr << re.what() << '\n';
    }

    return CRef;
}

bool DPCTLContext_AreEq(__dpctl_keep const DPCTLSyclContextRef CtxRef1,
                        __dpctl_keep const DPCTLSyclContextRef CtxRef2)
{
    if (!(CtxRef1 && CtxRef2))
        // \todo handle error
        return false;
    return (*unwrap(CtxRef1) == *unwrap(CtxRef2));
}

__dpctl_give DPCTLSyclContextRef
DPCTLContext_Copy(__dpctl_keep const DPCTLSyclContextRef CRef)
{
    auto Context = unwrap(CRef);
    if (!Context) {
        std::cerr << "Cannot copy DPCTLSyclContextRef as input is a nullptr\n";
        return nullptr;
    }
    try {
        auto CopiedContext = new context(*Context);
        return wrap(CopiedContext);
    } catch (std::bad_alloc const &ba) {
        // \todo log error
        std::cerr << ba.what() << '\n';
        return nullptr;
    }
}

__dpctl_give DPCTLDeviceVectorRef
DPCTLContext_GetDevices(__dpctl_keep const DPCTLSyclContextRef CRef)
{
    auto Context = unwrap(CRef);
    if (!Context) {
        std::cerr << "Can not retrieve devices from DPCTLSyclContextRef as "
                     "input is a nullptr\n";
        return nullptr;
    }
    vector_class<DPCTLSyclDeviceRef> *DevicesVectorPtr = nullptr;
    try {
        DevicesVectorPtr = new vector_class<DPCTLSyclDeviceRef>();
    } catch (std::bad_alloc const &ba) {
        // \todo log error
        std::cerr << ba.what() << '\n';
        return nullptr;
    }
    try {
        auto Devices = Context->get_devices();
        DevicesVectorPtr->reserve(Devices.size());
        for (const auto &Dev : Devices) {
            DevicesVectorPtr->emplace_back(wrap(new device(Dev)));
        }
        return wrap(DevicesVectorPtr);
    } catch (std::bad_alloc const &ba) {
        delete DevicesVectorPtr;
        // \todo log error
        std::cerr << ba.what() << '\n';
        return nullptr;
    } catch (const runtime_error &re) {
        delete DevicesVectorPtr;
        // \todo log error
        std::cerr << re.what() << '\n';
        return nullptr;
    }
}

size_t DPCTLContext_DeviceCount(__dpctl_keep const DPCTLSyclContextRef CRef)
{
    auto Context = unwrap(CRef);
    if (!Context) {
        std::cerr << "Can not retrieve devices from DPCTLSyclContextRef as "
                     "input is a nullptr\n";
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
        std::cerr << "Argument CtxRef is null"
                  << "/n";
        return 0;
    }
}
