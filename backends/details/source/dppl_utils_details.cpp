//===------ dppl_utils_details.cpp - dpctl-C_API  ----*---- C++ -----*-----===//
//
//               Data Parallel Control Library (dpCtl)
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
/// This file implements the helper functions defined in dppl_utils_details.h.
///
//===----------------------------------------------------------------------===//

#include "dppl_utils_details.h"
#include <string>
#include <sstream>

using namespace cl::sycl;

/*!
* Transforms enum info::device_type to string.
*/
std::string DDPL_StrToDeviceType(info::device_type devTy)
{
    std::stringstream ss;
    switch (devTy)
    {
    case info::device_type::cpu:
        ss << "cpu" << '\n';
        break;
    case info::device_type::gpu:
        ss << "gpu" << '\n';
        break;
    case info::device_type::accelerator:
        ss << "accelerator" << '\n';
        break;
    case info::device_type::custom:
        ss << "custom" << '\n';
        break;
    case info::device_type::host:
        ss << "host" << '\n';
        break;
    default:
        ss << "unknown" << '\n';
    }
    return ss.str();
}

/*!
* Transforms string to enum info::device_type.
*/
info::device_type DPPL_DeviceTypeToStr(std::string devTyStr)
{
    info::device_type devTy;
    if (devTyStr == "cpu") {
        devTy = info::device_type::cpu;
    } else if(devTyStr == "gpu") {
        devTy = info::device_type::gpu;
    } else if(devTyStr == "accelerator") {
        devTy = info::device_type::accelerator;
    } else if(devTyStr == "custom") {
        devTy = info::device_type::custom;
    } else if(devTyStr == "host") {
        devTy = info::device_type::host;
    } else {
        // \todo handle the error
        throw std::runtime_error("Unknown device type.");
    }
    return devTy;
}
