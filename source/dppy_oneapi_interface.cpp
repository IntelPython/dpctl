//===-- dppy_oneapi_interface.cpp - DPPY-SYCL interface ---*- C++ -*-------===//
//
//                     Data Parallel Python (DPPY)
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
#include <string>

using namespace cl::sycl;
using namespace dppy;

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

int64_t error_reporter (const std::string & msg)
{
    std::cerr << "Error: " << msg << '\n';
    return DPPY_FAILURE;
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
    active_contexts_.emplace_front(
        std::make_shared<DppyOneAPIContext>(default_selector())
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

    std::cout << "---Current DppyOneAPIContext :\n";
    active_contexts_.front()->dump();

    std::cout << "---Number of active DppyOneAPIContexts : "
              << active_contexts_.size() << '\n';

    return DPPY_SUCCESS;
}


int64_t DppyOneAPIRuntime::getNumPlatforms (size_t *platforms) const
{
    *platforms = num_platforms_;
    return DPPY_SUCCESS;
}


int64_t
DppyOneAPIRuntime::getCurrentContext (std::shared_ptr<DppyOneAPIContext> * Ctx)
const
{
    if (active_contexts_.empty())
        return error_reporter("No currently active context.");

    *Ctx = active_contexts_.front();
    return DPPY_SUCCESS;
}


int64_t
DppyOneAPIRuntime::setGlobalContextWithGPU (size_t DNum)
{
    if(active_contexts_.empty())
        return error_reporter("Why is there no previous global context?");
    if(DNum > gpu_devices_.size()) {
        std::stringstream ss;
        ss << "SYCL GPU device " << DNum << " not found on system.";
        return error_reporter(ss.str());
    }
    active_contexts_.pop_back();
    active_contexts_.push_back(std::make_shared<DppyOneAPIContext>(
                               gpu_devices_[DNum]));
    return DPPY_SUCCESS;
}


int64_t
DppyOneAPIRuntime::setGlobalContextWithCPU (size_t DNum)
{
    if(active_contexts_.empty())
        return error_reporter("Why is there no previous global context?");
    if(DNum > cpu_devices_.size()) {
        std::stringstream ss;
        ss << "SYCL CPU device " << DNum << " not found on system.";
        return error_reporter(ss.str());
    }
    active_contexts_.pop_back();
    active_contexts_.push_back(std::make_shared<DppyOneAPIContext>(
                               cpu_devices_[DNum]));
    return DPPY_SUCCESS;
}


int64_t
DppyOneAPIRuntime::pushGPUContext (std::shared_ptr<DppyOneAPIContext> * C,
                                   size_t DNum)
{
    if(DNum >= gpu_devices_.size()) {
        std::stringstream ss;
        ss << "SYCL GPU device " << DNum << " not found on system.";
        return error_reporter(ss.str());
    }

    active_contexts_.emplace_front(
        std::shared_ptr<DppyOneAPIContext>(
            new DppyOneAPIContext(gpu_devices_[DNum])
    ));
    *C = active_contexts_.front();
    return DPPY_SUCCESS;
}


int64_t
DppyOneAPIRuntime::pushCPUContext (std::shared_ptr<DppyOneAPIContext> * C,
                                   size_t device_num)
{
    if(device_num >= cpu_devices_.size()) {
        std::stringstream ss;
        ss << "SYCL CPU device " << device_num << " not found on system.";
        return error_reporter(ss.str());
    }

    active_contexts_.emplace_front(
        std::shared_ptr<DppyOneAPIContext>(
            new DppyOneAPIContext(cpu_devices_[device_num])
    ));
    *C = active_contexts_.front();
    return DPPY_SUCCESS;
}


int64_t DppyOneAPIRuntime::popContext ()
{
    if(active_contexts_.empty()) return error_reporter("No active contexts");
    active_contexts_.pop_front();
    return DPPY_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// DppyOneAPIContext ////////////////////////////
////////////////////////////////////////////////////////////////////////////////

DppyOneAPIContext::DppyOneAPIContext (const device_selector & DeviceSelector)
    : queue_(new queue(DeviceSelector))
{
}


DppyOneAPIContext::DppyOneAPIContext (const device & Device)
    : queue_(new queue(Device))
{
}


DppyOneAPIContext::DppyOneAPIContext (const DppyOneAPIContext & Ctx)
{
    queue_ = Ctx.queue_;
}


DppyOneAPIContext::DppyOneAPIContext (DppyOneAPIContext && Ctx)
{
    queue_ = Ctx.queue_;
}

DppyOneAPIContext& DppyOneAPIContext::operator=(const DppyOneAPIContext & Ctx)
{
    if (this != &Ctx) {
        queue_ = Ctx.queue_;
    }
    return *this;
}


DppyOneAPIContext& DppyOneAPIContext::operator=(DppyOneAPIContext && Ctx)
{
    if (this != &Ctx) {
        queue_ = std::move(Ctx.queue_);
    }
    return *this;
}

int64_t
DppyOneAPIContext::getSyclQueue (std::shared_ptr<cl::sycl::queue> * Queue) const
{
    if(!queue_.get()) return error_reporter("No valid queue exists.");
    *Queue = queue_;
    return DPPY_SUCCESS;
}


#if 0
int64_t DppyOneAPIContext::getSyclContext (cl::sycl::context * Context) const
{
    return DPPY_SUCCESS;
}


int64_t DppyOneAPIContext::getSyclDevice (cl::sycl::device * Device) const
{
    return DPPY_SUCCESS;
}


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
    if(!queue_)
        return DPPY_FAILURE;
    dump_device_info(queue_->get_device());
    return DPPY_SUCCESS;
}
