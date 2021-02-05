//===- dpctl_utils_helper.cpp - Implementation of enum to string helpers   ===//
//
//                      Data Parallel Control (dpCtl)
//
// Copyright 2020-2021 Intel Corporation
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
/// This file implements the helper functions defined in dpctl_utils_helper.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_utils_helper.h"
#include <sstream>
#include <string>

using namespace cl::sycl;

/*!
 * Transforms enum info::device_type to string.
 */
std::string DPCTL_DeviceTypeToStr(info::device_type devTy)
{
    std::stringstream ss;
    switch (devTy) {
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
info::device_type DPCTL_StrToDeviceType(const std::string & devTyStr)
{
    info::device_type devTy;
    if (devTyStr == "cpu") {
        devTy = info::device_type::cpu;
    }
    else if (devTyStr == "gpu") {
        devTy = info::device_type::gpu;
    }
    else if (devTyStr == "accelerator") {
        devTy = info::device_type::accelerator;
    }
    else if (devTyStr == "custom") {
        devTy = info::device_type::custom;
    }
    else if (devTyStr == "host") {
        devTy = info::device_type::host;
    }
    else {
        // \todo handle the error
        throw std::runtime_error("Unknown device type.");
    }
    return devTy;
}

backend DPCTL_DPCTLBackendTypeToSyclBackend(const DPCTLSyclBackendType & BeTy)
{
    switch(BeTy)
    {
        case DPCTLSyclBackendType::DPCTL_CUDA:
            return backend::cuda;
        case DPCTLSyclBackendType::DPCTL_HOST:
            return backend::host;
        case DPCTLSyclBackendType::DPCTL_LEVEL_ZERO:
            return backend::level_zero;
        case DPCTLSyclBackendType::DPCTL_OPENCL:
            return backend::opencl;
        default:
            throw runtime_error("Unsupported backend type", -1);
    }
}

info::device_type DPCTL_DPCTLDeviceTypeToSyclDeviceType (
    const DPCTLSyclDeviceType & DTy)
{
    switch(DTy)
    {
        case DPCTLSyclDeviceType::DPCTL_ACCELERATOR:
            return info::device_type::accelerator;
        case DPCTLSyclDeviceType::DPCTL_ALL:
            return info::device_type::all;
        case DPCTLSyclDeviceType::DPCTL_AUTOMATIC:
            return info::device_type::automatic;
        case DPCTLSyclDeviceType::DPCTL_CPU:
            return info::device_type::cpu;
        case DPCTLSyclDeviceType::DPCTL_CUSTOM:
            return info::device_type::custom;
        case DPCTLSyclDeviceType::DPCTL_GPU:
            return info::device_type::gpu;
        case DPCTLSyclDeviceType::DPCTL_HOST_DEVICE:
            return info::device_type::host;
        default:
            throw runtime_error("Unsupported device type", -1);
    }
}
