//===------ dppl_sycl_device_interface.cpp - dpctl-C_API  ---*--- C++ --*--===//
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
/// This file implements the data types and functions declared in
/// dppl_sycl_device_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dppl_sycl_device_interface.h"
#include "Support/CBindingWrapping.h"
#include <iomanip>
#include <iostream>
#include <cstring>
#include <CL/sycl.hpp>                /* SYCL headers   */
#include "../details/include/dppl_utils_details.h"

using namespace cl::sycl;

namespace
{
// Create wrappers for C Binding types (see CBindingWrapping.h).
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(device, DPPLSyclDeviceRef)

 /*!
 * @brief Helper function to print the metadata for a sycl::device.
 *
 * @param    Device         My Param doc
 */
void dump_device_info (const device & Device)
{
    std::stringstream ss;

    ss << std::setw(4) << " " << std::left << std::setw(16) << "Name"
       << Device.get_info<info::device::name>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(16) << "Driver version"
       << Device.get_info<info::device::driver_version>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(16) << "Vendor"
       << Device.get_info<info::device::vendor>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(16) << "Profile"
       << Device.get_info<info::device::profile>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(16) << "Device type";

    auto devTy = Device.get_info<info::device::device_type>();
    ss << ConvertEnumToStr(devTy);

    std::cout << ss.str();
}

} /* end of anonymous namespace */

/*!
 * Prints some of the device info metadata for the device corresponding to the
 * specified sycl::queue. Currently, device name, driver version, device
 * vendor, and device profile are printed out. More attributed may be added
 * later.
 */
void DPPLDevice_DumpInfo (__dppl_keep const DPPLSyclDeviceRef DRef)
{
    auto Device = unwrap(DRef);
    dump_device_info(*Device);
}


void DPPLDevice_Delete (__dppl_take DPPLSyclDeviceRef DRef)
{
    delete unwrap(DRef);
}

bool DPPLDevice_IsAccelerator (__dppl_keep const DPPLSyclDeviceRef DRef)
{
    auto D = unwrap(DRef);
    if (D) {
        return D->is_accelerator();
    }
    return false;
}

bool DPPLDevice_IsCPU (__dppl_keep const DPPLSyclDeviceRef DRef)
{
    auto D = unwrap(DRef);
    if (D) {
        return D->is_cpu();
    }
    return false;
}

bool DPPLDevice_IsGPU (__dppl_keep const DPPLSyclDeviceRef DRef)
{
    auto D = unwrap(DRef);
    if (D) {
        return D->is_gpu();
    }
    return false;
}


bool DPPLDevice_IsHost (__dppl_keep const DPPLSyclDeviceRef DRef)
{
    auto D = unwrap(DRef);
    if (D) {
        return D->is_host();
    }
    return false;
}


uint32_t
DPPLDevice_GetMaxComputeUnits (__dppl_keep const DPPLSyclDeviceRef DRef)
{
    auto D = unwrap(DRef);
    if (D) {
        return D->get_info<info::device::max_compute_units>();
    }
    return 0;
}

uint32_t
DPPLDevice_GetMaxWorkItemDims (__dppl_keep const DPPLSyclDeviceRef DRef)
{
    auto D = unwrap(DRef);
    if (D) {
        return D->get_info<info::device::max_work_item_dimensions>();
    }
    return 0;
}

__dppl_keep size_t*
DPPLDevice_GetMaxWorkItemSizes (__dppl_keep const DPPLSyclDeviceRef DRef)
{
    size_t *sizes = nullptr;
    auto D = unwrap(DRef);
    if (D) {
        auto id_sizes = D->get_info<info::device::max_work_item_sizes>();
        sizes = new size_t[3];
        for(auto i = 0ul; i < 3; ++i) {
            sizes[i] = id_sizes[i];
        }
    }
    return sizes;
}

size_t
DPPLDevice_GetMaxWorkGroupSize (__dppl_keep const DPPLSyclDeviceRef DRef)
{
    auto D = unwrap(DRef);
    if (D) {
        return D->get_info<info::device::max_work_group_size>();
    }
    return 0;
}

uint32_t
DPPLDevice_GetMaxNumSubGroups (__dppl_keep const DPPLSyclDeviceRef DRef)
{
    auto D = unwrap(DRef);
    if (D) {
        return D->get_info<info::device::max_num_sub_groups>();
    }
    return 0;
}

bool
DPPLDevice_HasInt64BaseAtomics (__dppl_keep const DPPLSyclDeviceRef DRef)
{
    auto D = unwrap(DRef);
    if (D) {
        return D->has(aspect::int64_base_atomics);
    }
    return false;
}

bool
DPPLDevice_HasInt64ExtendedAtomics (__dppl_keep const DPPLSyclDeviceRef DRef)
{
    auto D = unwrap(DRef);
    if (D) {
        return D->has(aspect::int64_extended_atomics);
    }
    return false;
}

__dppl_give const char*
DPPLDevice_GetName (__dppl_keep const DPPLSyclDeviceRef DRef)
{
    auto D = unwrap(DRef);
    if (D) {
        auto name = D->get_info<info::device::name>();
        auto cstr_len = name.length()+1;
        auto cstr_name = new char[cstr_len];
#ifdef _WIN32
        strncpy_s(cstr_name, cstr_len, name.c_str(), cstr_len);
#else
        std::strncpy(cstr_name, name.c_str(), cstr_len);
#endif
        return cstr_name;
    }
    return nullptr;
}

__dppl_give const char*
DPPLDevice_GetVendorName (__dppl_keep const DPPLSyclDeviceRef DRef)
{
    auto D = unwrap(DRef);
    if (D) {
        auto vendor = D->get_info<info::device::vendor>();
        auto cstr_len = vendor.length()+1;
        auto cstr_vendor = new char[cstr_len];
#ifdef _WIN32
        strncpy_s(cstr_vendor, cstr_len, vendor.c_str(), cstr_len);
#else
        std::strncpy(cstr_vendor, vendor.c_str(), cstr_len);
#endif
        return cstr_vendor;
    }
    return nullptr;
}

__dppl_give const char*
DPPLDevice_GetDriverInfo (__dppl_keep const DPPLSyclDeviceRef DRef)
{
    auto D = unwrap(DRef);
    if (D) {
        auto driver = D->get_info<info::device::driver_version>();
        auto cstr_len = driver.length()+1;
        auto cstr_driver = new char[cstr_len];
#ifdef _WIN32
        strncpy_s(cstr_driver, cstr_len, driver.c_str(), cstr_len);
#else
        std::strncpy(cstr_driver, driver.c_str(), cstr_len);
#endif
        return cstr_driver;
    }
    return nullptr;
}

bool DPPLDevice_IsHostUnifiedMemory (__dppl_keep const DPPLSyclDeviceRef DRef)
{
    auto D = unwrap(DRef);
    if (D) {
        return D->get_info<info::device::host_unified_memory>();
    }
    return false;
}

bool DPPLDevice_AreEq(__dppl_keep const DPPLSyclDeviceRef DevRef1,
                      __dppl_keep const DPPLSyclDeviceRef DevRef2)
{
    if(!(DevRef1 && DevRef2))
        // \todo handle error
        return false;
    return (*unwrap(DevRef1) == *unwrap(DevRef2));
}
