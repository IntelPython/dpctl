//=== dpctl_sycl_platform_manager.cpp - Implements helpers for sycl::platform //
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
/// dpctl_sycl_platform_manager.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_platform_manager.h"
#include "Support/CBindingWrapping.h"
#include "dpctl_error_handlers.h"
#include "dpctl_sycl_platform_interface.h"
#include "dpctl_utils_helper.h"
#include <CL/sycl.hpp>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>

using namespace cl::sycl;

namespace
{
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(platform, DPCTLSyclPlatformRef);

void platform_print_info_impl(const platform &p, size_t verbosity)
{
    std::stringstream ss;

    if (verbosity > 2) {
        error_handler("Illegal verbosity level. Accepted values are 0, 1, or 2."
                      "Defaulting to verbosity level 0.",
                      __FILE__, __func__, __LINE__);
        verbosity = 0;
    }

    if (verbosity == 0)
        ss << p.get_info<info::platform::name>() << " "
           << p.get_info<info::platform::version>() << '\n';

    if (verbosity > 0) {
        auto vendor = p.get_info<info::platform::vendor>();
        if (vendor.empty())
            vendor = "unknown";

        ss << std::setw(4) << " " << std::left << std::setw(12) << "Name"
           << p.get_info<info::platform::name>() << '\n'
           << std::setw(4) << " " << std::left << std::setw(12) << "Version"
           << p.get_info<info::platform::version>() << '\n'
           << std::setw(4) << " " << std::left << std::setw(12) << "Vendor"
           << vendor << '\n'
           << std::setw(4) << " " << std::left << std::setw(12) << "Backend";
        p.is_host() ? (ss << "unknown") : (ss << p.get_backend());
        ss << '\n';

        // Get number of devices on the platform
        auto devices = p.get_devices();
        ss << std::setw(4) << " " << std::left << std::setw(12) << "Num Devices"
           << devices.size() << '\n';

        if (verbosity == 2)
            // Print some of the device information
            for (auto dn = 0ul; dn < devices.size(); ++dn) {
                ss << std::setw(6) << " " << std::left << "# " << dn << '\n'
                   << std::setw(8) << " " << std::left << std::setw(20)
                   << "Name" << devices[dn].get_info<info::device::name>()
                   << '\n'
                   << std::setw(8) << " " << std::left << std::setw(20)
                   << "Version"
                   << devices[dn].get_info<info::device::driver_version>()
                   << '\n'
                   << std::setw(8) << " " << std::left << std::setw(20)
                   << "Filter string"
                   << devices[dn].get_platform().get_backend() << ":"
                   << DPCTL_DeviceTypeToStr(
                          devices[dn].get_info<info::device::device_type>())
                   << ":" << DPCTL_GetRelativeDeviceId(devices[dn]) << '\n';
            }
    }

    std::cout << ss.str();
}

} // namespace

#undef EL
#define EL Platform
#include "dpctl_vector_templ.cpp"
#undef EL

void DPCTLPlatformMgr_PrintInfo(__dpctl_keep const DPCTLSyclPlatformRef PRef,
                                size_t verbosity)
{
    auto p = unwrap(PRef);
    if (p) {
        platform_print_info_impl(*p, verbosity);
    }
    else {
        error_handler("Platform reference is NULL.", __FILE__, __func__,
                      __LINE__);
    }
}
