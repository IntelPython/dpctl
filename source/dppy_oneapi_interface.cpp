//===-- dppy_oneapi_interface.cpp - DPPY-SYCL interface ---*- C++ -*-------===//
//
//                     Data Parallel Python (DPPY)
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
using namespace dppy_rt;

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
/////////////////////////////// DppyOneAPIRuntime //////////////////////////////
////////////////////////////////////////////////////////////////////////////////

DppyOneAPIRuntime::DppyOneAPIRuntime ()
    :num_platforms_(platform::get_platforms().size()),
     cpu_contexts_(device::get_devices(info::device_type::cpu)),
     gpu_contexts_(device::get_devices(info::device_type::gpu))
{ }


DppyOneAPIRuntime::~DppyOneAPIRuntime ()
{

}

ErrorCode DppyOneAPIRuntime::dump () const
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
    if (cpu_contexts_.size())
        std::cout << "Number of available SYCL CPU devices: "
                  << cpu_contexts_.size() << '\n';

    // Print out the info for GPU devices
    if (gpu_contexts_.size())
        std::cout << "Number of available SYCL GPU devices: "
                  << gpu_contexts_.size() << '\n';

    return ErrorCode::DPPY_SUCCESS;
}

ErrorCode DppyOneAPIRuntime::getDefaultContext (DppyOneAPIContext *ctx) const
{
    if(available_contexts_.empty()) {
        std::cerr << "ERROR: Why are there no available contexts. There should "
                     "have been at least the default context.\n";
        return ErrorCode::DPPY_FAILURE;
    }

    // TODO copy stuff into ctx from back of deque

    return ErrorCode::DPPY_SUCCESS;
}

ErrorCode DppyOneAPIRuntime::setCurrentContext (info::device_type ty,
                                                size_t device_num)
{
    return ErrorCode::DPPY_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// DppyOneAPIContext ////////////////////////////
////////////////////////////////////////////////////////////////////////////////

auto DppyOneAPIContext::dump ()
{
    dump_device_info(queue_.get_device());
}
