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
#include <CL/sycl.hpp>                /* SYCL headers   */

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

    try {
        if (Device.has(aspect::accelerator))
            ss << "accelerator" << '\n';
        else if (Device.has(aspect::cpu))
            ss << "cpu" << '\n';
        else if (Device.has(aspect::custom))
            ss << "custom" << '\n';
        else if (Device.has(aspect::gpu))
            ss << "gpu" << '\n';
        else if (Device.has(aspect::host))
            ss << "host" << '\n';
    } catch (runtime_error re) {
        // \todo handle errors
        ss << "unknown\n";
    }

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
    return unwrap(DRef)->is_accelerator();
}

bool DPPLDevice_IsCPU (__dppl_keep const DPPLSyclDeviceRef DRef)
{
    return unwrap(DRef)->is_cpu();
}

bool DPPLDevice_IsGPU (__dppl_keep const DPPLSyclDeviceRef DRef)
{
    return unwrap(DRef)->is_gpu();
}


bool DPPLDevice_IsHost (__dppl_keep const DPPLSyclDeviceRef DRef)
{
    return unwrap(DRef)->is_host();
}

__dppl_give const char*
DPPLDevice_GetName (__dppl_keep const DPPLSyclDeviceRef DRef)
{
    auto name = unwrap(DRef)->get_info<info::device::name>();
    auto cstr_name = new char [name.length()+1];
    std::strcpy (cstr_name, name.c_str());
    return cstr_name;
}

__dppl_give const char*
DPPLDevice_GetVendorName (__dppl_keep const DPPLSyclDeviceRef DRef)
{
    auto vendor = unwrap(DRef)->get_info<info::device::name>();
    auto cstr_vendor = new char [vendor.length()+1];
    std::strcpy (cstr_vendor, vendor.c_str());
    return cstr_vendor;
}

__dppl_give const char*
DPPLDevice_GetDriverInfo (__dppl_keep const DPPLSyclDeviceRef DRef)
{
    auto driver = unwrap(DRef)->get_info<info::device::driver_version>();
    auto cstr_driver = new char [driver.length()+1];
    std::strcpy (cstr_driver, driver.c_str());
    return cstr_driver;
}

bool DPPLDevice_IsHostUnifiedMemory (__dppl_keep const DPPLSyclDeviceRef DRef)
{
    return unwrap(DRef)->get_info<info::device::host_unified_memory>();
}
