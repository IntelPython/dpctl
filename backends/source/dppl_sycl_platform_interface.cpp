//===--- dppl_sycl_platform_interface.cpp - DPPL-SYCL interface --*- C++ -*-===//
//
//               Python Data Parallel Processing Library (PyDPPL)
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
/// This file implements the functions declared in
/// dppl_sycl_platform_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dppl_sycl_platform_interface.h"
#include <iomanip>
#include <iostream>
#include <sstream>

#include <CL/sycl.hpp>

using namespace cl::sycl;

/*!
 * Prints out the following sycl::info::platform attributes for each platform
 * found on the system:
 *      - info::platform::name
 *      - info::platform::version
 *      - info::platform::profile
 *
 */
void DPPLPlatform_DumpInfo ()
{
    size_t i = 0;

    // Print out the info for each platform
    auto platforms = platform::get_platforms();
    for (auto &p : platforms) {
        std::cout << "---Platform " << i << '\n';
        std::stringstream ss;

        ss << std::setw(4) << " " << std::left << std::setw(12) << "Name"
           << p.get_info<info::platform::name>() << '\n';
        ss << std::setw(4) << " " << std::left << std::setw(12) << "Version"
           << p.get_info<info::platform::version>() << '\n';
        ss << std::setw(4) << " " << std::left << std::setw(12) << "Vendor"
           << p.get_info<info::platform::vendor>() << '\n';
        ss << std::setw(4) << " " << std::left << std::setw(12) << "Profile"
           << p.get_info<info::platform::profile>() << '\n';

        std::cout << ss.str();
        ++i;
    }
}

/*!
 * Returns the number of sycl::platform on the system.
 */
size_t DPPLPlatform_GetNumPlatforms ()
{
    return platform::get_platforms().size();
}
