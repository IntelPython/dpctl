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
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
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
////////////////////////////////// Free functions //////////////////////////////
////////////////////////////////////////////////////////////////////////////////


int64_t dppy::deleteQueue (void *Q)
{
    delete static_cast<queue*>(Q);
    return DPPY_SUCCESS;
}


////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// DppyOneAPIRuntime //////////////////////////////
////////////////////////////////////////////////////////////////////////////////


DppyOneAPIRuntime::DppyOneAPIRuntime ()
    : num_platforms_(platform::get_platforms().size())
{
    for(auto d : device::get_devices(info::device_type::cpu))
        cpu_queues_.emplace_back(d);
    for(auto d : device::get_devices(info::device_type::gpu))
        gpu_queues_.emplace_back(d);

    active_queues_.emplace_back(default_selector());
}


DppyOneAPIRuntime::~DppyOneAPIRuntime ()
{

}


int64_t DppyOneAPIRuntime::dump_queue (const queue *Queue)  const
{
    dump_device_info(Queue->get_device());
    return DPPY_SUCCESS;
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
    if (cpu_queues_.size())
        std::cout << "---Number of available SYCL CPU queues: "
                  << cpu_queues_.size() << '\n';
    else
        std::cout << "---No available SYCL CPU device\n";

    // Print out the info for GPU devices
    if (gpu_queues_.size())
        std::cout << "---Number of available SYCL GPU queues: "
                  << gpu_queues_.size() << '\n';
    else
        std::cout << "---No available SYCL GPU device\n";

    std::cout << "---Current queue :\n";
    dump_queue(&active_queues_.front());

    std::cout << "---Number of active queues : "
              << active_queues_.size() << '\n';

    return DPPY_SUCCESS;
}


int64_t DppyOneAPIRuntime::getNumPlatforms (size_t *platforms) const
{
    *platforms = num_platforms_;
    return DPPY_SUCCESS;
}


int64_t DppyOneAPIRuntime::getCurrentQueue (queue **Q) const
{
    if (active_queues_.empty())
        return error_reporter("No currently active queues.");
    *Q = new queue(active_queues_.front());
    return DPPY_SUCCESS;
}


int64_t DppyOneAPIRuntime::getQueue (queue **Q, info::device_type DeviceTy,
                                     size_t DNum) const
{
    if (DeviceTy == info::device_type::cpu) {
        try {
            *Q = new queue(cpu_queues_.at(DNum));
        }
        catch (const std::out_of_range& e) {
            std::stringstream ss;
            ss << "SYCL CPU device " << DNum << " not found on system.";
            return error_reporter(ss.str());
        }
        return DPPY_SUCCESS;
    }
    else if (DeviceTy == info::device_type::gpu) {
        try {
            *Q = new queue(gpu_queues_.at(DNum));
        }
        catch (const std::out_of_range& e) {
            std::stringstream ss;
            ss << "SYCL GPU device " << DNum << " not found on system.";
            return error_reporter(ss.str());
        }
        return DPPY_SUCCESS;
    }
    else {
        return error_reporter("Unsupported device type.");
    }
}


int64_t DppyOneAPIRuntime::resetGlobalQueue (info::device_type DeviceTy,
                                             size_t DNum)
{
    if(active_queues_.empty())
        return error_reporter("Why is there no previous global context?");

    // Remove the previous global queue, which is the first queue added
    // to the deque when the Runtime is initialized
    active_queues_.pop_back();

    switch (DeviceTy)
    {
    case info::device_type::cpu:
    {
        try {
            active_queues_.push_back(cpu_queues_.at(DNum));
        }
        catch (const std::out_of_range& e) {
            std::stringstream ss;
            ss << "SYCL CPU device " << DNum << " not found on system.";
            return error_reporter(ss.str());
        }
        break;
    }
    case info::device_type::gpu:
    {
        try {
            active_queues_.push_back(gpu_queues_.at(DNum));
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

    return DPPY_SUCCESS;
}


int64_t
DppyOneAPIRuntime::activateQueue (cl::sycl::queue **Q,
                                  cl::sycl::info::device_type DeviceTy,
                                  size_t DNum)
{
    if(active_queues_.empty())
        return error_reporter("Why is there no previous global context?");

    switch (DeviceTy)
    {
    case info::device_type::cpu:
    {
        try {
            active_queues_.emplace_front(cpu_queues_.at(DNum));
        }
        catch (const std::out_of_range& e) {
            std::stringstream ss;
            ss << "SYCL CPU device " << DNum << " not found on system.";
            return error_reporter(ss.str());
        }
        break;
    }
    case info::device_type::gpu:
    {
        try {
            active_queues_.emplace_front(gpu_queues_.at(DNum));
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

    *Q = new queue(active_queues_.front());
    return DPPY_SUCCESS;
}


int64_t DppyOneAPIRuntime::deactivateCurrentQueue ()
{
    if(active_queues_.empty()) return error_reporter("No active contexts");
    active_queues_.pop_front();
    return DPPY_SUCCESS;
}


////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// DppyOneAPIBuffer /////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename T>
DppyOneAPIBuffer<T>::DppyOneAPIBuffer (T* hostData, size_t ndims,
                                       const size_t dims[],
                                       const property_list & propList)
    : ndims_(ndims)
{
    switch (ndims)
    {
    case 1:
    {
        auto r = range(dims[0]);
        buff_ = buffer<T>(hostData, r, propList);
        break;
    }
    case 2:
    {
        auto r = range<2>(dims[0], dims[1]);
        buff_ = buffer<T>(hostData, r, propList);
        break;
    }
    case 3:
    {
        auto r = range<3>(dims[0], dims[1], dims[2]);
        buff_ = buffer<T>(hostData, r, propList);
        break;
    }
    default:
        throw std::invalid_argument("number of dimensions cannot be "
                                    "greater than three");
    }

    dims_ = new size_t[ndims];
    std::memmove(dims_, dims, ndims * sizeof(size_t));
}


template <typename T>
DppyOneAPIBuffer<T>::DppyOneAPIBuffer(const T* hostData, size_t ndims,
                                      const size_t dims[],
                                      const property_list& propList)
    : ndims_(ndims)
{
    switch (ndims)
    {
    case 1:
    {
        auto r = range(dims[0]);
        buff_ = buffer<T>(hostData, r, propList);
        break;
    }
    case 2:
    {
        auto r = range<2>(dims[0], dims[1]);
        buff_ = buffer<T>(hostData, r, propList);
        break;
    }
    case 3:
    {
        auto r = range<3>(dims[0], dims[1], dims[2]);
        buff_ = buffer<T>(hostData, r, propList);
        break;
    }
    default:
        throw std::invalid_argument("number of dimensions cannot be "
                                    "greater than three");
    }
    dims_ = new size_t[ndims];
    std::memmove(dims_, dims, ndims * sizeof(size_t));
}


template <typename T>
DppyOneAPIBuffer<T>::DppyOneAPIBuffer(size_t ndims, const size_t dims[],
                                      const property_list& propList)
    : ndims_(ndims)
{
    switch (ndims)
    {
    case 1:
    {
        auto r = range(dims[0]);
        buff_ = buffer<T>(r, propList);
        break;
    }
    case 2:
    {
        auto r = range<2>(dims[0], dims[1]);
        buff_ = buffer<T>(r, propList);
        break;
    }
    case 3:
    {
        auto r = range<3>(dims[0], dims[1], dims[2]);
        buff_ = buffer<T>(r, propList);
        break;
    }
    default:
        throw std::invalid_argument("number of dimensions cannot be "
                                    "greater than three");
    }
    dims_ = new size_t[ndims];
    std::memmove(dims_, dims, ndims * sizeof(size_t));
}


template <typename T>
DppyOneAPIBuffer<T>::~DppyOneAPIBuffer()
{
    delete dims_;
}
