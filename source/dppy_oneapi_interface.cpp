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
    default_context_ = std::make_shared<DppyOneAPIContext>(default_selector());
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
    default_context_->dump();

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
DppyOneAPIRuntime::getDefaultContext (std::shared_ptr<DppyOneAPIContext> * Ctx)
const
{
    *Ctx = default_context_;
    return DPPY_SUCCESS;
}


int64_t
DppyOneAPIRuntime::setDefaultContext (std::shared_ptr<DppyOneAPIContext> * Ctx,
                                      info::device_type DTy,
                                      size_t DNum)
{
    switch(DTy)
    {
    case info::device_type::gpu:
        if(DNum > gpu_devices_.size()) {
            std::stringstream ss;
            ss << "SYCL GPU device " << DNum << " not found on system.";
            return error_reporter(ss.str());
        }
        default_context_ = std::make_shared<DppyOneAPIContext>(
                               gpu_devices_[DNum]);
        *Ctx = default_context_;
        break;
    case info::device_type::cpu:
        if(DNum > cpu_devices_.size()) {
            std::stringstream ss;
            ss << "SYCL CPU device " << DNum << " not found on system.";
            return error_reporter(ss.str());
        }
        default_context_ = std::make_shared<DppyOneAPIContext>(
                               cpu_devices_[DNum]);
        *Ctx = default_context_;
        break;
    default:
        break;
    }
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
        delete queue_;
        queue_ = Ctx.queue_;
    }
    return *this;
}

DppyOneAPIContext& DppyOneAPIContext::operator=(DppyOneAPIContext && Ctx)
{
    std::cout << "Move assign...\n";
    if (this != &Ctx) {
        delete queue_;
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
    if(!queue_)
        return DPPY_FAILURE;
    dump_device_info(queue_->get_device());
    return DPPY_SUCCESS;
}

DppyOneAPIContext::~DppyOneAPIContext ()
{
    delete queue_;
}
