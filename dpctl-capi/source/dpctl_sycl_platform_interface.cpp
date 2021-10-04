//=== dpctl_sycl_platform_interface.cpp - Implements C API for sycl::platform //
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
/// dpctl_sycl_platform_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_platform_interface.h"
#include "../helper/include/dpctl_utils_helper.h"
#include "Support/CBindingWrapping.h"
#include <CL/sycl.hpp>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <vector>

using namespace cl::sycl;

namespace
{
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(platform, DPCTLSyclPlatformRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(device_selector, DPCTLSyclDeviceSelectorRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(std::vector<DPCTLSyclPlatformRef>,
                                   DPCTLPlatformVectorRef);
} // namespace

__dpctl_give DPCTLSyclPlatformRef
DPCTLPlatform_Copy(__dpctl_keep const DPCTLSyclPlatformRef PRef)
{
    auto Platform = unwrap(PRef);
    if (!Platform) {
        std::cerr << "Cannot copy DPCTLSyclPlatformRef as input is a nullptr\n";
        return nullptr;
    }
    try {
        auto CopiedPlatform = new platform(*Platform);
        return wrap(CopiedPlatform);
    } catch (std::bad_alloc const &ba) {
        // \todo log error
        std::cerr << ba.what() << '\n';
        return nullptr;
    }
}

__dpctl_give DPCTLSyclPlatformRef DPCTLPlatform_Create()
{
    DPCTLSyclPlatformRef PRef = nullptr;
    try {
        auto P = new platform();
        PRef = wrap(P);
    } catch (std::bad_alloc const &ba) {
        std::cerr << ba.what() << '\n';
    }
    return PRef;
}

__dpctl_give DPCTLSyclPlatformRef DPCTLPlatform_CreateFromSelector(
    __dpctl_keep const DPCTLSyclDeviceSelectorRef DSRef)
{
    if (DSRef) {
        auto DS = unwrap(DSRef);
        platform *P = nullptr;
        try {
            P = new platform(*DS);
            return wrap(P);
        } catch (std::bad_alloc const &ba) {
            std::cerr << ba.what() << '\n';
            return nullptr;
        } catch (runtime_error const &re) {
            delete P;
            std::cerr << re.what() << '\n';
            return nullptr;
        }
    }
    else {
        std::cerr << "Device selector pointer cannot be NULL\n";
    }

    return nullptr;
}

void DPCTLPlatform_Delete(__dpctl_take DPCTLSyclPlatformRef PRef)
{
    auto P = unwrap(PRef);
    delete P;
}

DPCTLSyclBackendType
DPCTLPlatform_GetBackend(__dpctl_keep const DPCTLSyclPlatformRef PRef)
{
    DPCTLSyclBackendType BTy = DPCTLSyclBackendType::DPCTL_UNKNOWN_BACKEND;
    auto P = unwrap(PRef);
    if (P) {
        BTy = DPCTL_SyclBackendToDPCTLBackendType(P->get_backend());
    }
    else {
        std::cerr << "Backend cannot be looked up for a NULL platform\n";
    }
    return BTy;
}

__dpctl_give const char *
DPCTLPlatform_GetName(__dpctl_keep const DPCTLSyclPlatformRef PRef)
{
    char *cstr_name = nullptr;
    auto P = unwrap(PRef);
    if (P) {
        try {
            auto name = P->get_info<info::platform::name>();
            auto cstr_len = name.length() + 1;
            cstr_name = new char[cstr_len];
#ifdef _WIN32
            strncpy_s(cstr_name, cstr_len, name.c_str(), cstr_len);
#else
            std::strncpy(cstr_name, name.c_str(), cstr_len);
#endif
        } catch (std::bad_alloc const &ba) {
            // \todo log error
            std::cerr << ba.what() << '\n';
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    else {
        std::cerr << "Name cannot be looked up for a NULL platform\n";
    }
    return cstr_name;
}

__dpctl_give const char *
DPCTLPlatform_GetVendor(__dpctl_keep const DPCTLSyclPlatformRef PRef)
{
    char *cstr_vendor = nullptr;
    auto P = unwrap(PRef);
    if (P) {
        try {
            auto vendor = P->get_info<info::platform::vendor>();
            auto cstr_len = vendor.length() + 1;
            cstr_vendor = new char[cstr_len];
#ifdef _WIN32
            strncpy_s(cstr_vendor, cstr_len, vendor.c_str(), cstr_len);
#else
            std::strncpy(cstr_vendor, vendor.c_str(), cstr_len);
#endif
        } catch (std::bad_alloc const &ba) {
            // \todo log error
            std::cerr << ba.what() << '\n';
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    else {
        std::cerr << "Vendor cannot be looked up for a NULL platform\n";
    }
    return cstr_vendor;
}

__dpctl_give const char *
DPCTLPlatform_GetVersion(__dpctl_keep const DPCTLSyclPlatformRef PRef)
{
    char *cstr_driver = nullptr;
    auto P = unwrap(PRef);
    if (P) {
        try {
            auto driver = P->get_info<info::platform::version>();
            auto cstr_len = driver.length() + 1;
            cstr_driver = new char[cstr_len];
#ifdef _WIN32
            strncpy_s(cstr_driver, cstr_len, driver.c_str(), cstr_len);
#else
            std::strncpy(cstr_driver, driver.c_str(), cstr_len);
#endif
        } catch (std::bad_alloc const &ba) {
            // \todo log error
            std::cerr << ba.what() << '\n';
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    else {
        std::cerr << "Driver version cannot be looked up for a NULL platform\n";
    }
    return cstr_driver;
}

__dpctl_give DPCTLPlatformVectorRef DPCTLPlatform_GetPlatforms()
{
    std::vector<DPCTLSyclPlatformRef> *Platforms = nullptr;

    auto platforms = platform::get_platforms();

    try {
        Platforms = new std::vector<DPCTLSyclPlatformRef>();
        Platforms->reserve(platforms.size());
    } catch (std::bad_alloc const &ba) {
        return nullptr;
    }

    // populate the vector
    for (const auto &P : platforms) {
        Platforms->emplace_back(wrap(new platform(P)));
    }

    // the wrap function is defined inside dpctl_vector_templ.cpp
    return wrap(Platforms);
}
