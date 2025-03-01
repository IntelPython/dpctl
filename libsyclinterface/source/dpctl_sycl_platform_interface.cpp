//=== dpctl_sycl_platform_interface.cpp - Implements C API for sycl::platform //
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2025 Intel Corporation
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
/// dpctl_sycl_platform_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_platform_interface.h"
#include "Config/dpctl_config.h"
#include "dpctl_device_selection.hpp"
#include "dpctl_error_handlers.h"
#include "dpctl_string_utils.hpp"
#include "dpctl_sycl_enum_types.h"
#include "dpctl_sycl_type_casters.hpp"
#include "dpctl_utils_helper.h"
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
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

platform *new_platform_from_selector(const dpctl_device_selector *sel)
{
    return new platform(
        [=](const device &d) -> int { return sel->operator()(d); });
}

} // end of anonymous namespace

__dpctl_give DPCTLSyclPlatformRef
DPCTLPlatform_Copy(__dpctl_keep const DPCTLSyclPlatformRef PRef)
{
    auto Platform = unwrap<platform>(PRef);
    if (!Platform) {
        error_handler("Cannot copy DPCTLSyclPlatformRef as input is a nullptr.",
                      __FILE__, __func__, __LINE__);
        return nullptr;
    }
    try {
        auto CopiedPlatform = new platform(*Platform);
        return wrap<platform>(CopiedPlatform);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

__dpctl_give DPCTLSyclPlatformRef DPCTLPlatform_Create()
{
    DPCTLSyclPlatformRef PRef = nullptr;
    try {
        auto P = new platform();
        PRef = wrap<platform>(P);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
    }
    return PRef;
}

__dpctl_give DPCTLSyclPlatformRef DPCTLPlatform_CreateFromSelector(
    __dpctl_keep const DPCTLSyclDeviceSelectorRef DSRef)
{
    if (DSRef) {
        auto DS = unwrap<dpctl_device_selector>(DSRef);
        platform *P = nullptr;
        try {
            P = new_platform_from_selector(DS);
            return wrap<platform>(P);
        } catch (std::exception const &e) {
            delete P;
            error_handler(e, __FILE__, __func__, __LINE__);
            return nullptr;
        }
    }
    else {
        error_handler("Device selector pointer cannot be NULL.", __FILE__,
                      __func__, __LINE__);
    }

    return nullptr;
}

void DPCTLPlatform_Delete(__dpctl_take DPCTLSyclPlatformRef PRef)
{
    auto P = unwrap<platform>(PRef);
    delete P;
}

DPCTLSyclBackendType
DPCTLPlatform_GetBackend(__dpctl_keep const DPCTLSyclPlatformRef PRef)
{
    DPCTLSyclBackendType BTy = DPCTLSyclBackendType::DPCTL_UNKNOWN_BACKEND;
    auto P = unwrap<platform>(PRef);
    if (P) {
        BTy = DPCTL_SyclBackendToDPCTLBackendType(P->get_backend());
    }
    else {
        error_handler("Backend cannot be looked up for a NULL platform.",
                      __FILE__, __func__, __LINE__);
    }
    return BTy;
}

__dpctl_give const char *
DPCTLPlatform_GetName(__dpctl_keep const DPCTLSyclPlatformRef PRef)
{
    auto P = unwrap<platform>(PRef);
    if (P) {
        try {
            auto name = P->get_info<info::platform::name>();
            return dpctl::helper::cstring_from_string(name);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
            return nullptr;
        }
    }
    else {
        error_handler("Name cannot be looked up for a NULL platform.", __FILE__,
                      __func__, __LINE__);
        return nullptr;
    }
}

__dpctl_give const char *
DPCTLPlatform_GetVendor(__dpctl_keep const DPCTLSyclPlatformRef PRef)
{
    auto P = unwrap<platform>(PRef);
    if (P) {
        try {
            auto vendor = P->get_info<info::platform::vendor>();
            return dpctl::helper::cstring_from_string(vendor);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
            return nullptr;
        }
    }
    else {
        error_handler("Vendor cannot be looked up for a NULL platform.",
                      __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

__dpctl_give const char *
DPCTLPlatform_GetVersion(__dpctl_keep const DPCTLSyclPlatformRef PRef)
{
    auto P = unwrap<platform>(PRef);
    if (P) {
        try {
            auto driver = P->get_info<info::platform::version>();
            return dpctl::helper::cstring_from_string(driver);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
            return nullptr;
        }
    }
    else {
        error_handler("Driver version cannot be looked up for a NULL platform.",
                      __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

__dpctl_give DPCTLPlatformVectorRef DPCTLPlatform_GetPlatforms()
{
    using vecTy = std::vector<DPCTLSyclPlatformRef>;
    vecTy *Platforms = nullptr;

    std::vector<platform> platforms;
    try {
        platforms = platform::get_platforms();
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }

    try {
        Platforms = new vecTy();
        Platforms->reserve(platforms.size());
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }

    // populate the vector
    for (const auto &P : platforms) {
        Platforms->emplace_back(wrap<platform>(new platform(P)));
    }

    // the wrap function is defined inside dpctl_vector_templ.cpp
    return wrap<vecTy>(Platforms);
}

__dpctl_give DPCTLSyclContextRef
DPCTLPlatform_GetDefaultContext(__dpctl_keep const DPCTLSyclPlatformRef PRef)
{
    auto P = unwrap<platform>(PRef);
    if (P) {
#ifdef SYCL_EXT_ONEAPI_DEFAULT_CONTEXT
        try {
            const auto &default_ctx = P->ext_oneapi_get_default_context();
            return wrap<context>(new context(default_ctx));
        } catch (const std::exception &ex) {
            error_handler(ex, __FILE__, __func__, __LINE__);
            return nullptr;
        }
#else
        return nullptr;
#endif
    }
    else {
        error_handler(
            "Default platform cannot be obtained up for a NULL platform.",
            __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

bool DPCTLPlatform_AreEq(__dpctl_keep const DPCTLSyclPlatformRef PRef1,
                         __dpctl_keep const DPCTLSyclPlatformRef PRef2)
{
    auto P1 = unwrap<platform>(PRef1);
    auto P2 = unwrap<platform>(PRef2);
    if (P1 && P2)
        return *P1 == *P2;
    else
        return false;
}

size_t DPCTLPlatform_Hash(__dpctl_keep const DPCTLSyclPlatformRef PRef)
{
    if (PRef) {
        auto P = unwrap<platform>(PRef);
        std::hash<platform> hash_fn;
        return hash_fn(*P);
    }
    else {
        error_handler("Argument PRef is null.", __FILE__, __func__, __LINE__);
        return 0;
    }
}

__dpctl_give DPCTLDeviceVectorRef
DPCTLPlatform_GetDevices(__dpctl_keep const DPCTLSyclPlatformRef PRef,
                         DPCTLSyclDeviceType DTy)
{
    auto P = unwrap<platform>(PRef);
    if (!P) {
        error_handler("Cannot retrieve devices from DPCTLSyclPlatformRef as "
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

    // handle unknown device
    if (DTy == DPCTLSyclDeviceType::DPCTL_UNKNOWN_DEVICE) {
        return wrap<vecTy>(DevicesVectorPtr);
    }

    try {
        auto SyclDTy = DPCTL_DPCTLDeviceTypeToSyclDeviceType(DTy);
        auto Devices = P->get_devices(SyclDTy);
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
