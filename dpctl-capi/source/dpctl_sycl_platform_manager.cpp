//=== dpctl_sycl_platform_manager.cpp - Implements helpers for sycl::platform //
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
/// This file implements the functions declared in
/// dpctl_sycl_platform_manager.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_platform_manager.h"
#include "../helper/include/dpctl_utils_helper.h"
#include "Support/CBindingWrapping.h"
#include "dpctl_sycl_platform_interface.h"
#include <CL/sycl.hpp>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>

using namespace cl::sycl;

namespace
{
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(platform, DPCTLSyclPlatformRef);

void platform_print_info_impl(const platform &p)
{
    std::stringstream ss;

    auto vendor = p.get_info<info::platform::vendor>();
    if (vendor.empty())
        vendor = "unknown";

    ss << std::setw(4) << " " << std::left << std::setw(12) << "Name"
       << p.get_info<info::platform::name>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(12) << "Version"
       << p.get_info<info::platform::version>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(12) << "Vendor"
       << vendor << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(12) << "Profile"
       << p.get_info<info::platform::profile>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(12) << "Backend";
    p.is_host() ? (ss << "unknown") : (ss << p.get_backend());
    ss << '\n';

    // Get number of devices on the platform
    auto devices = p.get_devices();

    ss << std::setw(4) << " " << std::left << std::setw(12) << "Devices"
       << devices.size() << '\n';
    // Print some of the device information
    for (auto dn = 0ul; dn < devices.size(); ++dn) {
        ss << std::setw(6) << " " << std::left << std::setw(12) << "Device "
           << dn << '\n';
        ss << std::setw(8) << " " << std::left << std::setw(20) << "Name"
           << devices[dn].get_info<info::device::name>() << '\n';
        ss << std::setw(8) << " " << std::left << std::setw(20)
           << "Driver version"
           << devices[dn].get_info<info::device::driver_version>() << '\n';
        ss << std::setw(8) << " " << std::left << std::setw(20)
           << "Device type";

        auto devTy = devices[dn].get_info<info::device::device_type>();
        ss << DPCTL_DeviceTypeToStr(devTy);
    }
    std::cout << ss.str();
}

} // namespace

#undef EL
#define EL Platform
#include "dpctl_vector_templ.cpp"
#undef EL

/*!
 * Prints out the following sycl::info::platform attributes for the platform:
 *      - info::platform::name
 *      - info::platform::version
 *      - info::platform::vendor
 *      - info::platform::profile
 *      - backend (opencl, cuda, level-zero, host)
 *      - number of devices on the platform
 *
 * Additionally, for each device associated with the platform print out:
 *      - info::device::name
 *      - info::device::driver_version
 *      - type of the device based on the aspects cpu, gpu, accelerator.
 */
void DPCTLPlatformMgr_PrintInfo(__dpctl_keep const DPCTLSyclPlatformRef PRef)
{
    auto p = unwrap(PRef);
    if (p) {
        platform_print_info_impl(*p);
    }
    else {
        std::cout << "Platform reference is NULL.\n";
    }
}
