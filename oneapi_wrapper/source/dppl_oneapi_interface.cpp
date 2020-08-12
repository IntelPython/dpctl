//===-- dppl_oneapi_interface.cpp - DPPL-SYCL interface ---*- C++ -*-------===//
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
/// This file implements the data types and functions declared in
/// dppl_oneapi_interface.hpp.
///
//===----------------------------------------------------------------------===//
#include "dppl_oneapi_interface.hpp"
#include "dppl_error_codes.hpp"
#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <CL/sycl.hpp>                /* SYCL headers   */

using namespace cl::sycl;
using namespace dppl;

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
    return DPPL_FAILURE;
}

class DpplOneAPIRuntimeHelper
{
    size_t                                  num_platforms_;
    cl::sycl::vector_class<cl::sycl::queue> cpu_queues_;
    cl::sycl::vector_class<cl::sycl::queue> gpu_queues_;
    std::deque<cl::sycl::queue>             active_queues_;
public:
    DpplOneAPIRuntimeHelper ()
        : num_platforms_(platform::get_platforms().size())
    {
        for(auto d : device::get_devices(info::device_type::cpu))
            cpu_queues_.emplace_back(d);
        for(auto d : device::get_devices(info::device_type::gpu))
            gpu_queues_.emplace_back(d);

        active_queues_.emplace_back(default_selector());
    }

    ~DpplOneAPIRuntimeHelper ()
    {

    }

    friend dppl::DpplOneAPIRuntime;
};

// This singleton function is needed to create the DpplOneAPIRuntimeHelper object 
//  in a predictable manner without which there is a chance of segfault.
DpplOneAPIRuntimeHelper& get_gRtHelper()
{
    static DpplOneAPIRuntimeHelper * helper = new DpplOneAPIRuntimeHelper();
    return *helper;
}

#define gRtHelper get_gRtHelper()

} /* end of anonymous namespace */


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// Free functions //////////////////////////////
////////////////////////////////////////////////////////////////////////////////


int64_t dppl::deleteQueue (void *Q)
{
    delete static_cast<queue*>(Q);
    return DPPL_SUCCESS;
}


////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// DpplOneAPIRuntime //////////////////////////////
////////////////////////////////////////////////////////////////////////////////


int64_t DpplOneAPIRuntime::dump_queue (const void *QPtr) const
{
    auto Q = static_cast<const queue*>(QPtr);
    dump_device_info(Q->get_device());
    return DPPL_SUCCESS;
}

int64_t DpplOneAPIRuntime::dump () const
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
    if (gRtHelper.cpu_queues_.size())
        std::cout << "---Number of available SYCL CPU queues: "
                  << gRtHelper.cpu_queues_.size() << '\n';
    else
        std::cout << "---No available SYCL CPU device\n";

    // Print out the info for GPU devices
    if (gRtHelper.gpu_queues_.size())
        std::cout << "---Number of available SYCL GPU queues: "
                  << gRtHelper.gpu_queues_.size() << '\n';
    else
        std::cout << "---No available SYCL GPU device\n";

    std::cout << "---Current queue :\n";
    dump_queue(&gRtHelper.active_queues_.front());

    std::cout << "---Number of active queues : "
              << gRtHelper.active_queues_.size() << '\n';

    return DPPL_SUCCESS;
}


int64_t DpplOneAPIRuntime::getNumPlatforms (size_t *platforms) const
{
    *platforms = gRtHelper.num_platforms_;
    return DPPL_SUCCESS;
}


int64_t DpplOneAPIRuntime::getCurrentQueue (void **QPtr) const
{
    if (gRtHelper.active_queues_.empty())
        return error_reporter("No currently active queues.");

    *QPtr = new queue(gRtHelper.active_queues_.front());
    return DPPL_SUCCESS;
}


int64_t DpplOneAPIRuntime::getQueue (void **QPtr, sycl_device_type DeviceTy,
                                     size_t DNum) const
{
    if (DeviceTy == sycl_device_type::cpu) {
        try {
            *QPtr = new queue(gRtHelper.cpu_queues_.at(DNum));
        }
        catch (const std::out_of_range& e) {
            std::stringstream ss;
            ss << "SYCL CPU device " << DNum << " not found on system.";
            return error_reporter(ss.str());
        }
        return DPPL_SUCCESS;
    }
    else if (DeviceTy == sycl_device_type::gpu) {
        try {
            *QPtr = new queue(gRtHelper.gpu_queues_.at(DNum));
        }
        catch (const std::out_of_range& e) {
            std::stringstream ss;
            ss << "SYCL GPU device " << DNum << " not found on system.";
            return error_reporter(ss.str());
        }
        return DPPL_SUCCESS;
    }
    else {
        return error_reporter("Unsupported device type.");
    }
}


int64_t DpplOneAPIRuntime::resetGlobalQueue (sycl_device_type DeviceTy,
                                             size_t DNum)
{
    if(gRtHelper.active_queues_.empty())
        return error_reporter("Why is there no previous global context?");

    // Remove the previous global queue, which if never previously reset will
    // be the first queue that was added to the deque when the Runtime was
    // initialized.
    gRtHelper.active_queues_.pop_back();

    switch (DeviceTy)
    {
    case sycl_device_type::cpu:
    {
        try {
            gRtHelper.active_queues_.push_back(gRtHelper.cpu_queues_.at(DNum));
        }
        catch (const std::out_of_range& e) {
            std::stringstream ss;
            ss << "SYCL CPU device " << DNum << " not found on system.";
            return error_reporter(ss.str());
        }
        break;
    }
    case sycl_device_type::gpu:
    {
        try {
            gRtHelper.active_queues_.push_back(gRtHelper.gpu_queues_.at(DNum));
        }
        catch (const std::out_of_range& e) {
            std::stringstream ss;
            ss << "SYCL GPU device " << DNum << " not found on system.";
            return error_reporter(ss.str());
        }
        break;
    }
    default:
    {
        return error_reporter("Unsupported device type.");
    }
    }

    return DPPL_SUCCESS;
}


int64_t
DpplOneAPIRuntime::activateQueue (void **QPtr,
                                  sycl_device_type DeviceTy,
                                  size_t DNum)
{
    if(gRtHelper.active_queues_.empty())
        return error_reporter("Why is there no previous global context?");

    switch (DeviceTy)
    {
    case sycl_device_type::cpu:
    {
        try {
            gRtHelper.active_queues_.emplace_front(
                gRtHelper.cpu_queues_.at(DNum)
            );
        }
        catch (const std::out_of_range& e) {
            std::stringstream ss;
            ss << "SYCL CPU device " << DNum << " not found on system.";
            return error_reporter(ss.str());
        }
        break;
    }
    case sycl_device_type::gpu:
    {
        try {
            gRtHelper.active_queues_.emplace_front(
                gRtHelper.gpu_queues_.at(DNum)
            );
        }
        catch (const std::out_of_range& e) {
            std::stringstream ss;
            ss << "SYCL GPU device " << DNum << " not found on system.";
            return error_reporter(ss.str());
        }
        break;
    }
    default:
    {
        return error_reporter("Unsupported device type.");
    }
    }

    *QPtr = new queue(gRtHelper.active_queues_.front());
    return DPPL_SUCCESS;
}


int64_t DpplOneAPIRuntime::deactivateCurrentQueue ()
{
    if(gRtHelper.active_queues_.empty())
        return error_reporter("No active contexts");
    gRtHelper.active_queues_.pop_front();
    return DPPL_SUCCESS;
}


int64_t DpplOneAPIRuntime::number_of_activated_queues (size_t &num)
{
    if (gRtHelper.active_queues_.empty())
        return error_reporter("No default queue present.");
    num = gRtHelper.active_queues_.size()-1;
    return DPPL_SUCCESS;
}
