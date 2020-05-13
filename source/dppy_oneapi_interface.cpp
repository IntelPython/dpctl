//===-- dppy_oneapi_interface.cpp - DPPY-SYCL interface ---*- C++ -*-------===//
//
//                     Data Parallel Python (DPPY)
//
// This file is distributed under the University of Illinois Open Source
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
/// This file implements the the data types and functions declared in
/// dppy_oneapi_interface.h
///
//===----------------------------------------------------------------------===//
#include "dppy_oneapi_interface.hpp"
#include "error_check_macros.h"
#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace cl::sycl;

/*------------------------------- Private helpers ----------------------------*/

// Anonymous namespace for private helpers
namespace
{

void dump_device_info (const device & Device)
{
    std::stringstream ss;

    ss << std::setw(4) << " " << std::left << std::setw(12) << "Name"
       << Device.get_info<info::device::name>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(12) << "Driver version"
       << Device.get_info<info::device::driver_version>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(12) << "Vendor"
       << Device.get_info<info::device::vendor>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(12) << "Profile"
       << Device.get_info<info::device::profile>() << '\n';

    std::cout << ss.str();
}

void dump_platform_info (const platform & Platform)
{
    std::stringstream ss;

    ss << std::setw(4) << " " << std::left << std::setw(12) << "Name"
       << Platform.get_info<info::platform::name>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(12) << "Version"
       << Platform.get_info<info::platform::version>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(12) << "Vendor"
       << Platform.get_info<info::platform::vendor>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(12) << "Profile"
       << Platform.get_info<info::platform::profile>() << '\n';

    std::cout << ss.str();
}

} /* end of anonymous namespace */

////////////////////////////////////////////////////////////////////////////////
//////////////////////////// DppyOneAPIContextFactory //////////////////////////
////////////////////////////////////////////////////////////////////////////////

DppyOneAPIContextFactory::DppyOneAPIContextFactory ()
    :num_platforms_(platform::get_platforms().size()),
     num_cpus_(device::get_devices(info::device_type::cpu).size()),
     num_gpus_(device::get_devices(info::device_type::gpu).size())
{ }


auto DppyOneAPIContextFactory::dump () const
{
    size_t i = 0;

    // Print out the info for each platform
    auto platforms = platform::get_platforms();
    for (auto &p : platforms) {
        std::cout << "Platform " << i << '\n';
        dump_platform_info(p);
        ++i;
    }

    // Print out the info for CPU devices
    if(num_cpus_)
        std::cout << "Number of available SYCL CPU devices: "
                  << num_cpus_ << '\n';

    // Print out the info for GPU devices
    if(num_gpus_)
        std::cout << "Number of available SYCL GPU devices: "
                  << num_gpus_ << '\n';
}

////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// DppyOneAPIContext ////////////////////////////
////////////////////////////////////////////////////////////////////////////////

auto DppyOneAPIContext::dump ()
{
    dump_device_info(queue_->get_device());
}
