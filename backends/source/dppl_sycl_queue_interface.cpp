//===--- dppl_sycl_queue_interface.cpp - DPPL-SYCL interface --*- C++ -*---===//
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
/// dppl_sycl_queue_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dppl_sycl_queue_interface.h"
#include "Support/CBindingWrapping.h"
#include <iomanip>
#include <iostream>
#include <sstream>

#include <CL/sycl.hpp>                /* SYCL headers   */

using namespace cl::sycl;

namespace
{
// Create wrappers for C Binding types (see CBindingWrapping.h).
 DEFINE_SIMPLE_CONVERSION_FUNCTIONS(queue, DPPLSyclQueueRef)

/*!
 * @brief
 *
 * @param    Platform       My Param doc
 */
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

/*!
 * Prints out number of available SYCL platforms, number of CPU queues, number
 * of GPU queues, metadata about the current global queue, and how many queues
 * are currently activated. More information can be added in future, and
 * functions to extract these information using SYCL API (e.g. device_info)
 * may also be added. For now, this function can be used as a basic canary test
 * to check if the queue manager was properly initialized.
 *
 */
void DPPLDumpPlatformInfo ()
{
    size_t i = 0;

    // Print out the info for each platform
    auto platforms = platform::get_platforms();
    for (auto &p : platforms) {
        std::cout << "---Platform " << i << '\n';
        dump_platform_info(p);
        ++i;
    }
}

__dppl_give DPPLSyclDeviceRef
DPPLGetDeviceFromQueue (__dppl_keep const DPPLSyclQueueRef QRef)
{
    auto Q = unwrap(QRef);
    auto Device = new device(Q->get_device());
    return reinterpret_cast<DPPLSyclDeviceRef>(Device);
}

__dppl_give DPPLSyclContextRef
DPPLGetContextFromQueue (__dppl_keep const DPPLSyclQueueRef QRef)
{
    auto Q = unwrap(QRef);
    auto Context = new context(Q->get_context());
    return reinterpret_cast<DPPLSyclContextRef>(Context);
}

/*!
 * Delete the passed in pointer after verifying it points to a sycl::queue.
 */
void DPPLDeleteSyclQueue (__dppl_take DPPLSyclQueueRef QRef)
{
    delete unwrap(QRef);
}
