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

    ss << std::setw(4) << " " << std::left << std::setw(16) << "Name"
       << Device.get_info<info::device::name>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(16) << "Driver version"
       << Device.get_info<info::device::driver_version>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(16) << "Vendor"
       << Device.get_info<info::device::vendor>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(16) << "Profile"
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
    : num_platforms_(platform::get_platforms().size()),
      cpu_devices_(device::get_devices(info::device_type::cpu)),
      gpu_devices_(device::get_devices(info::device_type::gpu))
{
    contexts_.emplace_back(
        std::make_shared<DppyOneAPIContext>(default_selector())
        //std::shared_ptr<DppyOneAPIContext>(
        //    new DppyOneAPIContext(default_selector()))
    );
}


DppyOneAPIRuntime::~DppyOneAPIRuntime ()
{

}


int64_t DppyOneAPIRuntime::dump () const
{
    size_t i = 0;

    // Print out the info for each platform
    auto platforms = platform::get_platforms();
    for (auto &p : platforms) {
        std::cout << "---Platform " << i << '\n';
        dump_platform_info(p);
        ++i;
    }

    // Print out the info for CPU devices
    if (cpu_devices_.size())
        std::cout << "---Number of available SYCL CPU devices: "
                  << cpu_devices_.size() << '\n';
    else
        std::cout << "---No available SYCL CPU device\n";

    // Print out the info for GPU devices
    if (gpu_devices_.size())
        std::cout << "---Number of available SYCL GPU devices: "
                  << gpu_devices_.size() << '\n';
    else
        std::cout << "---No available SYCL GPU device\n";

    std::cout << "---Default DppyOneAPIContext initialized to device :\n";
    contexts_.front()->dump();

    return DPPY_SUCCESS;
}

int64_t
DppyOneAPIRuntime::getCurrentContext (std::shared_ptr<DppyOneAPIContext> & Ctx)
const
{
    if(contexts_.empty()) {
        std::cerr << "ERROR: Why are there no available contexts. There should "
                     "have been at least the default context.\n";
        return DPPY_FAILURE;
    }

    if(Ctx.get() != nullptr) {
        std::cerr << "ERROR: Context argument needs to be uninitialized\n";
        return DPPY_FAILURE;
    }

    Ctx = contexts_.front();
    return DPPY_SUCCESS;
}


int64_t DppyOneAPIRuntime::setCurrentContext (info::device_type Ty,
                                                size_t device_num)
{
    switch(Ty)
    {
    case info::device_type::gpu:
        if (device_num < gpu_devices_.size())
            contexts_.emplace_front(
                std::shared_ptr<DppyOneAPIContext>(
                    new DppyOneAPIContext(default_selector())
            ));
        else {
            std::cerr << "ERROR: SYCL GPU device " << device_num
                      << " does not exist.\n";
            return DPPY_FAILURE;
        }
        break;
    case info::device_type::cpu:
        if (device_num < cpu_devices_.size())
            contexts_.emplace_front(
                std::shared_ptr<DppyOneAPIContext>(
                    new DppyOneAPIContext(default_selector())
            ));
        else {
            std::cerr << "ERROR: SYCL CPU device " << device_num
                      << " does not exist.\n";
            return DPPY_FAILURE;
        }
        break;
        break;
    default:
        std::cerr << "ERROR: Device type not currently supported.\n";
        return DPPY_FAILURE;
        break;
    }
    return DPPY_SUCCESS;
}


int64_t DppyOneAPIRuntime::resetCurrentContext ()
{
    if(contexts_.size() > 1) {
        std::cerr << "ERROR: Resetting current context would leave no "
                     "usable context.\n";
        return DPPY_FAILURE;
    }
    contexts_.pop_front();
    return DPPY_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// DppyOneAPIContext ////////////////////////////
////////////////////////////////////////////////////////////////////////////////

DppyOneAPIContext::DppyOneAPIContext (const device_selector & DeviceSelector)
    : queue_(DeviceSelector)
{
}

DppyOneAPIContext::DppyOneAPIContext (const device & Device)
    : queue_(Device)
{
}

DppyOneAPIContext::DppyOneAPIContext (const DppyOneAPIContext & Ctx)
    : queue_(Ctx.queue_)
{
    std::cout << "Copy...\n";
}

DppyOneAPIContext::DppyOneAPIContext (DppyOneAPIContext && Ctx)
{
    queue_ = Ctx.queue_;
    std::cout << "Move...\n";
}

DppyOneAPIContext& DppyOneAPIContext::operator=(const DppyOneAPIContext & Ctx)
{
    std::cout << "Copy assign...\n";
    if (this != &Ctx) {
       queue_ = Ctx.queue_;
    }
    return *this;
}

DppyOneAPIContext& DppyOneAPIContext::operator=(DppyOneAPIContext && Ctx)
{
    std::cout << "Move assign...\n";

    if (this != &Ctx) {
        queue_ = std::move(Ctx.queue_);
    }
    return *this;
}

int64_t DppyOneAPIContext::getSyclQueue (cl::sycl::queue * Queue) const
{
    return DPPY_SUCCESS;
}

int64_t DppyOneAPIContext::getSyclContext (cl::sycl::context * Context) const
{
    return DPPY_SUCCESS;
}

int64_t DppyOneAPIContext::getSyclDevice (cl::sycl::device * Device) const
{
    return DPPY_SUCCESS;
}

#if 0
int64_t DppyOneAPIContext::getOpenCLQueue () const
{

}

int64_t DppyOneAPIContext::getOpenCLContext () const
{

}

int64_t DppyOneAPIContext::getOpenCLDevice () const
{

}
#endif

int64_t DppyOneAPIContext::dump ()
{
    dump_device_info(queue_.get_device());
    return DPPY_SUCCESS;
}
