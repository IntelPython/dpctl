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
info::device_type DPCTL_StrToDeviceType(const std::string &devTyStr)
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

backend DPCTL_DPCTLBackendTypeToSyclBackend(DPCTLSyclBackendType BeTy)
{
    switch (BeTy) {
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

DPCTLSyclBackendType DPCTL_SyclBackendToDPCTLBackendType(backend B)
{
    switch (B) {
    case backend::cuda:
        return DPCTLSyclBackendType::DPCTL_CUDA;
    case backend::host:
        return DPCTLSyclBackendType::DPCTL_HOST;
    case backend::level_zero:
        return DPCTLSyclBackendType::DPCTL_LEVEL_ZERO;
    case backend::opencl:
        return DPCTLSyclBackendType::DPCTL_OPENCL;
    default:
        return DPCTLSyclBackendType::DPCTL_UNKNOWN_BACKEND;
    }
}

info::device_type DPCTL_DPCTLDeviceTypeToSyclDeviceType(DPCTLSyclDeviceType DTy)
{
    switch (DTy) {
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

DPCTLSyclDeviceType DPCTL_SyclDeviceTypeToDPCTLDeviceType(info::device_type D)
{
    switch (D) {
    case info::device_type::accelerator:
        return DPCTLSyclDeviceType::DPCTL_ACCELERATOR;
    case info::device_type::all:
        return DPCTLSyclDeviceType::DPCTL_ALL;
    case info::device_type::automatic:
        return DPCTLSyclDeviceType::DPCTL_AUTOMATIC;
    case info::device_type::cpu:
        return DPCTLSyclDeviceType::DPCTL_CPU;
    case info::device_type::custom:
        return DPCTLSyclDeviceType::DPCTL_CUSTOM;
    case info::device_type::gpu:
        return DPCTLSyclDeviceType::DPCTL_GPU;
    case info::device_type::host:
        return DPCTLSyclDeviceType::DPCTL_HOST_DEVICE;
    default:
        return DPCTLSyclDeviceType::DPCTL_UNKNOWN_DEVICE;
    }
}

/*!
 * Transforms cl::sycl::aspect to string.
 */
std::string DPCTL_AspectToStr(aspect aspectTy)
{
    std::stringstream ss;
    switch (aspectTy) {
    case aspect::host:
        ss << "host" << '\n';
        break;
    case aspect::cpu:
        ss << "cpu" << '\n';
        break;
    case aspect::gpu:
        ss << "gpu" << '\n';
        break;
    case aspect::accelerator:
        ss << "accelerator" << '\n';
        break;
    case aspect::custom:
        ss << "custom" << '\n';
        break;
    case aspect::fp16:
        ss << "fp16" << '\n';
        break;
    case aspect::fp64:
        ss << "fp64" << '\n';
        break;
    case aspect::int64_base_atomics:
        ss << "int64_base_atomics" << '\n';
        break;
    case aspect::int64_extended_atomics:
        ss << "int64_extended_atomics" << '\n';
        break;
    case aspect::image:
        ss << "image" << '\n';
        break;
    case aspect::online_compiler:
        ss << "online_compiler" << '\n';
        break;
    case aspect::online_linker:
        ss << "online_linker" << '\n';
        break;
    case aspect::queue_profiling:
        ss << "queue_profiling" << '\n';
        break;
    case aspect::usm_device_allocations:
        ss << "usm_device_allocations" << '\n';
        break;
    case aspect::usm_host_allocations:
        ss << "usm_host_allocations" << '\n';
        break;
    case aspect::usm_shared_allocations:
        ss << "usm_shared_allocations" << '\n';
        break;
    case aspect::usm_restricted_shared_allocations:
        ss << "usm_restricted_shared_allocations" << '\n';
        break;
    case aspect::usm_system_allocator:
        ss << "usm_system_allocator" << '\n';
        break;
    default:
        throw runtime_error("Unsupported aspect type", -1);
    }
    return ss.str();
}

/*!
 * Transforms string to cl::sycl::aspect.
 */
aspect DPCTL_StrToAspectType(const std::string &aspectTyStr)
{
    aspect aspectTy;
    if (aspectTyStr == "host") {
        aspectTy = aspect::host;
    }
    else if (aspectTyStr == "cpu") {
        aspectTy = aspect::cpu;
    }
    else if (aspectTyStr == "gpu") {
        aspectTy = aspect::gpu;
    }
    else if (aspectTyStr == "accelerator") {
        aspectTy = aspect::accelerator;
    }
    else if (aspectTyStr == "custom") {
        aspectTy = aspect::custom;
    }
    else if (aspectTyStr == "fp16") {
        aspectTy = aspect::fp16;
    }
    else if (aspectTyStr == "fp64") {
        aspectTy = aspect::fp64;
    }
    else if (aspectTyStr == "int64_base_atomics") {
        aspectTy = aspect::int64_base_atomics;
    }
    else if (aspectTyStr == "int64_extended_atomics") {
        aspectTy = aspect::int64_extended_atomics;
    }
    else if (aspectTyStr == "image") {
        aspectTy = aspect::image;
    }
    else if (aspectTyStr == "online_compiler") {
        aspectTy = aspect::online_compiler;
    }
    else if (aspectTyStr == "online_linker") {
        aspectTy = aspect::online_linker;
    }
    else if (aspectTyStr == "queue_profiling") {
        aspectTy = aspect::queue_profiling;
    }
    else if (aspectTyStr == "usm_device_allocations") {
        aspectTy = aspect::usm_device_allocations;
    }
    else if (aspectTyStr == "usm_host_allocations") {
        aspectTy = aspect::usm_host_allocations;
    }
    else if (aspectTyStr == "usm_shared_allocations") {
        aspectTy = aspect::usm_shared_allocations;
    }
    else if (aspectTyStr == "usm_restricted_shared_allocations") {
        aspectTy = aspect::usm_restricted_shared_allocations;
    }
    else if (aspectTyStr == "usm_system_allocator") {
        aspectTy = aspect::usm_system_allocator;
    }
    else {
        // \todo handle the error
        throw runtime_error("Unsupported aspect type", -1);
    }
    return aspectTy;
}

aspect DPCTL_DPCTLAspectTypeToSyclAspect(DPCTLSyclAspectType AspectTy)
{
    switch (AspectTy) {
    case DPCTLSyclAspectType::host:
        return aspect::host;
    case DPCTLSyclAspectType::cpu:
        return aspect::cpu;
    case DPCTLSyclAspectType::gpu:
        return aspect::gpu;
    case DPCTLSyclAspectType::accelerator:
        return aspect::accelerator;
    case DPCTLSyclAspectType::custom:
        return aspect::custom;
    case DPCTLSyclAspectType::fp16:
        return aspect::fp16;
    case DPCTLSyclAspectType::fp64:
        return aspect::fp64;
    case DPCTLSyclAspectType::int64_base_atomics:
        return aspect::int64_base_atomics;
    case DPCTLSyclAspectType::int64_extended_atomics:
        return aspect::int64_extended_atomics;
    case DPCTLSyclAspectType::image:
        return aspect::image;
    case DPCTLSyclAspectType::online_compiler:
        return aspect::online_compiler;
    case DPCTLSyclAspectType::online_linker:
        return aspect::online_linker;
    case DPCTLSyclAspectType::queue_profiling:
        return aspect::queue_profiling;
    case DPCTLSyclAspectType::usm_device_allocations:
        return aspect::usm_device_allocations;
    case DPCTLSyclAspectType::usm_host_allocations:
        return aspect::usm_host_allocations;
    case DPCTLSyclAspectType::usm_shared_allocations:
        return aspect::usm_shared_allocations;
    case DPCTLSyclAspectType::usm_restricted_shared_allocations:
        return aspect::usm_restricted_shared_allocations;
    case DPCTLSyclAspectType::usm_system_allocator:
        return aspect::usm_system_allocator;
    default:
        throw runtime_error("Unsupported aspect type", -1);
    }
}

DPCTLSyclAspectType DPCTL_SyclAspectToDPCTLAspectType(aspect Aspect)
{
    switch (Aspect) {
    case aspect::host:
        return DPCTLSyclAspectType::host;
    case aspect::cpu:
        return DPCTLSyclAspectType::cpu;
    case aspect::gpu:
        return DPCTLSyclAspectType::gpu;
    case aspect::accelerator:
        return DPCTLSyclAspectType::accelerator;
    case aspect::custom:
        return DPCTLSyclAspectType::custom;
    case aspect::fp16:
        return DPCTLSyclAspectType::fp16;
    case aspect::fp64:
        return DPCTLSyclAspectType::fp64;
    case aspect::int64_base_atomics:
        return DPCTLSyclAspectType::int64_base_atomics;
    case aspect::int64_extended_atomics:
        return DPCTLSyclAspectType::int64_extended_atomics;
    case aspect::image:
        return DPCTLSyclAspectType::image;
    case aspect::online_compiler:
        return DPCTLSyclAspectType::online_compiler;
    case aspect::online_linker:
        return DPCTLSyclAspectType::online_linker;
    case aspect::queue_profiling:
        return DPCTLSyclAspectType::queue_profiling;
    case aspect::usm_device_allocations:
        return DPCTLSyclAspectType::usm_device_allocations;
    case aspect::usm_host_allocations:
        return DPCTLSyclAspectType::usm_host_allocations;
    case aspect::usm_shared_allocations:
        return DPCTLSyclAspectType::usm_shared_allocations;
    case aspect::usm_restricted_shared_allocations:
        return DPCTLSyclAspectType::usm_restricted_shared_allocations;
    case aspect::usm_system_allocator:
        return DPCTLSyclAspectType::usm_system_allocator;
    default:
        throw runtime_error("Unsupported aspect type", -1);
    }
}