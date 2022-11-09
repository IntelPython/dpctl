//=== dpctl_sycl_platform_manager.cpp - Implements helpers for sycl::platform //
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2022 Intel Corporation
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
#include "Config/dpctl_config.h"
#include "dpctl_error_handlers.h"
#include "dpctl_string_utils.hpp"
#include "dpctl_sycl_platform_interface.h"
#include "dpctl_sycl_type_casters.hpp"
#include "dpctl_utils_helper.h"
#include <CL/sycl.hpp>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>

using namespace sycl;

namespace
{

using namespace dpctl::syclinterface;

std::string platform_print_info_impl(const platform &p, size_t verbosity)
{
    std::stringstream ss;
    static constexpr const char *_endl = "\n";

    if (verbosity > 2) {
        error_handler("Illegal verbosity level. Accepted values are 0, 1, or 2."
                      "Defaulting to verbosity level 0.",
                      __FILE__, __func__, __LINE__);
        verbosity = 0;
    }

    if (verbosity == 0)
        ss << p.get_info<info::platform::name>() << " "
           << p.get_info<info::platform::version>() << _endl;

    if (verbosity > 0) {
        auto vendor = p.get_info<info::platform::vendor>();
        if (vendor.empty())
            vendor = "unknown";

        ss << std::setw(4) << " " << std::left << std::setw(12) << "Name"
           << p.get_info<info::platform::name>() << _endl << std::setw(4) << " "
           << std::left << std::setw(12) << "Version"
           << p.get_info<info::platform::version>() << _endl << std::setw(4)
           << " " << std::left << std::setw(12) << "Vendor" << vendor << _endl
           << std::setw(4) << " " << std::left << std::setw(12) << "Backend";
#if __SYCL_COMPILER_VERSION >= __SYCL_COMPILER_2023_SWITCHOVER
        ss << p.get_backend();
#else
        p.is_host() ? (ss << "unknown") : (ss << p.get_backend());
#endif
        ss << _endl;

        // Get number of devices on the platform
        auto devices = p.get_devices();
        ss << std::setw(4) << " " << std::left << std::setw(12) << "Num Devices"
           << devices.size() << _endl;

        if (verbosity == 2)
            // Print some of the device information
            for (auto dn = 0ul; dn < devices.size(); ++dn) {
                ss << std::setw(6) << " " << std::left << "# " << dn << _endl
                   << std::setw(8) << " " << std::left << std::setw(20)
                   << "Name" << devices[dn].get_info<info::device::name>()
                   << _endl << std::setw(8) << " " << std::left << std::setw(20)
                   << "Version"
                   << devices[dn].get_info<info::device::driver_version>()
                   << _endl << std::setw(8) << " " << std::left << std::setw(20)
                   << "Filter string"
                   << DPCTL_GetDeviceFilterString(devices[dn]) << _endl;
            }
    }

    return ss.str();
}

} // namespace

#undef EL
#undef EL_SYCL_TYPE
#define EL Platform
#define EL_SYCL_TYPE sycl::platform
#include "dpctl_vector_templ.cpp"
#undef EL
#undef EL_SYCL_TYPE

void DPCTLPlatformMgr_PrintInfo(__dpctl_keep const DPCTLSyclPlatformRef PRef,
                                size_t verbosity)
{
    auto p = unwrap<platform>(PRef);
    if (p) {
        std::cout << platform_print_info_impl(*p, verbosity);
    }
    else {
        error_handler("Platform reference is NULL.", __FILE__, __func__,
                      __LINE__);
    }
}

__dpctl_give const char *
DPCTLPlatformMgr_GetInfo(__dpctl_keep const DPCTLSyclPlatformRef PRef,
                         size_t verbosity)
{
    const char *cstr_info = nullptr;
    auto p = unwrap<platform>(PRef);
    if (p) {
        auto infostr = platform_print_info_impl(*p, verbosity);
        cstr_info = dpctl::helper::cstring_from_string(infostr);
    }
    else {
        error_handler("Platform reference is NULL.", __FILE__, __func__,
                      __LINE__);
    }
    return cstr_info;
}
