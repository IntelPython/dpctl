//===-------- dpctl_sycl_device_manager.cpp - helpers for sycl devices ------=//
//
//                      Data Parallel Control (dpCtl)
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
/// This file implements the functions declared in dpctl_sycl_device_manager.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_device_manager.h"
#include "../helper/include/dpctl_utils_helper.h"
#include "Support/CBindingWrapping.h"
#include "dpctl_sycl_enum_types.h"
#include <CL/sycl.hpp> /* SYCL headers   */
#include <iomanip>
#include <iostream>

using namespace cl::sycl;

namespace
{

// Create wrappers for C Binding types (see CBindingWrapping.h).
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(device, DPCTLSyclDeviceRef)

/*!
 * @brief Helper function to print the metadata for a sycl::device.
 *
 * @param    Device         My Param doc
 */
void print_device_info(const device &Device)
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
    ss << DPCTL_DeviceTypeToStr(devTy);

    std::cout << ss.str();
}

template <backend Bty, info::device_type Dty> const size_t &getNumDevices()
{
    auto get_num_devices = [] {
        size_t ndevices = 0ul;
        auto Platforms = platform::get_platforms();
        for (auto const &P : Platforms) {
            if (P.is_host())
                continue;
            auto be = P.get_backend();
            if (be == Bty) {
                auto Devices = P.get_devices();
                for (auto &Device : Devices) {
                    auto devty = Device.get_info<info::device::device_type>();
                    if (devty == Dty)
                        ++ndevices;
                }
            }
        }
        return ndevices;
    };

    static size_t ndevices = get_num_devices();
    return ndevices;
}

template <backend Bty,
          info::device_type Dty,
          typename std::enable_if<Bty != backend::host &&
                                  Dty == info::device_type::all>::type>
const size_t &getNumDevices()
{
    auto get_num_devices = [] {
        size_t ndevices = 0ul;
        auto Platforms = platform::get_platforms();
        for (auto const &P : Platforms) {
            if (P.is_host())
                continue;
            auto be = P.get_backend();
            if (be == Bty)
                ndevices = P.get_devices().size();
        }
        return ndevices;
    };

    static size_t ndevices = get_num_devices();
    return ndevices;
}

} // namespace

/*!
 * Returns the number of available devices for a specific backend and device
 * type combination.
 */
size_t DPCTLDeviceMgr_GetNumDevices(int device_identifier)
{
    size_t nDevices = 0;

    if (device_identifier & DPCTL_CUDA ||
        (device_identifier & (DPCTL_CUDA | DPCTL_ALL)))
    {
        nDevices = getNumDevices<backend::cuda, info::device_type::all>();
    }
    else if (device_identifier & DPCTL_HOST ||
             (device_identifier & (DPCTL_HOST | DPCTL_ALL)) ||
             (device_identifier & (DPCTL_HOST | DPCTL_HOST_DEVICE)))
    {
        nDevices = 1;
    }
    else if (device_identifier & DPCTL_LEVEL_ZERO ||
             (device_identifier & (DPCTL_LEVEL_ZERO | DPCTL_ALL)))
    {
        nDevices = getNumDevices<backend::level_zero, info::device_type::all>();
    }
    else if (device_identifier & DPCTL_OPENCL ||
             (device_identifier & (DPCTL_OPENCL | DPCTL_ALL)))
    {
        nDevices = getNumDevices<backend::opencl, info::device_type::all>();
    }
    else if (device_identifier & (DPCTL_CUDA | DPCTL_GPU)) {
        nDevices = getNumDevices<backend::cuda, info::device_type::gpu>();
    }
    else if (device_identifier & (DPCTL_LEVEL_ZERO | DPCTL_GPU)) {
        nDevices = getNumDevices<backend::level_zero, info::device_type::gpu>();
    }
    else if (device_identifier & (DPCTL_OPENCL | DPCTL_GPU)) {
        nDevices = getNumDevices<backend::opencl, info::device_type::gpu>();
    }
    else if (device_identifier & (DPCTL_OPENCL | DPCTL_CPU)) {
        nDevices = getNumDevices<backend::opencl, info::device_type::cpu>();
    }

    return nDevices;
}

/*!
 * Prints some of the device info metadata for the device corresponding to the
 * specified sycl::queue. Currently, device name, driver version, device
 * vendor, and device profile are printed out. More attributed may be added
 * later.
 */
void DPCTLDeviceMgr_PrintDeviceInfo(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    auto Device = unwrap(DRef);
    print_device_info(*Device);
}