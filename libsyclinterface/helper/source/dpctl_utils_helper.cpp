//===- dpctl_utils_helper.cpp - Implementation of enum to string helpers   ===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2022 Intel Corporation
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
#include "Config/dpctl_config.h"
#include <sstream>
#include <string>

using namespace sycl;

/*!
 * Transforms enum info::device_type to string.
 */
std::string DPCTL_DeviceTypeToStr(info::device_type devTy)
{
    std::stringstream ss;
    switch (devTy) {
    case info::device_type::cpu:
        ss << "cpu";
        break;
    case info::device_type::gpu:
        ss << "gpu";
        break;
    case info::device_type::accelerator:
        ss << "accelerator";
        break;
    case info::device_type::custom:
        ss << "custom";
        break;
#if __SYCL_COMPILER_VERSION < __SYCL_COMPILER_2023_SWITCHOVER
    case info::device_type::host:
        ss << "host";
        break;
#endif
    default:
        ss << "unknown";
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
        return backend::ext_oneapi_cuda;
#if __SYCL_COMPILER_VERSION < __SYCL_COMPILER_2023_SWITCHOVER
    case DPCTLSyclBackendType::DPCTL_HOST:
        return backend::host;
#endif
    case DPCTLSyclBackendType::DPCTL_LEVEL_ZERO:
        return backend::ext_oneapi_level_zero;
    case DPCTLSyclBackendType::DPCTL_OPENCL:
        return backend::opencl;
    case DPCTLSyclBackendType::DPCTL_ALL_BACKENDS:
        return backend::all;
    default:
        throw std::runtime_error("Unsupported backend type");
    }
}

DPCTLSyclBackendType DPCTL_SyclBackendToDPCTLBackendType(backend B)
{
    switch (B) {
    case backend::ext_oneapi_cuda:
        return DPCTLSyclBackendType::DPCTL_CUDA;
#if __SYCL_COMPILER_VERSION < __SYCL_COMPILER_2023_SWITCHOVER
    case backend::host:
        return DPCTLSyclBackendType::DPCTL_HOST;
#endif
    case backend::ext_oneapi_level_zero:
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
        throw std::runtime_error("Unsupported device type");
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
 * Transforms sycl::aspect to string.
 */
std::string DPCTL_AspectToStr(aspect aspectTy)
{
    std::stringstream ss;
    switch (aspectTy) {
#if __SYCL_COMPILER_VERSION < __SYCL_COMPILER_2023_SWITCHOVER
    case aspect::host:
        ss << "host";
        break;
#endif
    case aspect::cpu:
        ss << "cpu";
        break;
    case aspect::gpu:
        ss << "gpu";
        break;
    case aspect::accelerator:
        ss << "accelerator";
        break;
    case aspect::custom:
        ss << "custom";
        break;
    case aspect::fp16:
        ss << "fp16";
        break;
    case aspect::fp64:
        ss << "fp64";
        break;
    case aspect::atomic64:
        ss << "atomic64";
        break;
    case aspect::image:
        ss << "image";
        break;
    case aspect::online_compiler:
        ss << "online_compiler";
        break;
    case aspect::online_linker:
        ss << "online_linker";
        break;
    case aspect::queue_profiling:
        ss << "queue_profiling";
        break;
    case aspect::usm_device_allocations:
        ss << "usm_device_allocations";
        break;
    case aspect::usm_host_allocations:
        ss << "usm_host_allocations";
        break;
    case aspect::usm_shared_allocations:
        ss << "usm_shared_allocations";
        break;
    case aspect::usm_restricted_shared_allocations:
        ss << "usm_restricted_shared_allocations";
        break;
    case aspect::usm_system_allocations:
        ss << "usm_system_allocations";
        break;
    case aspect::usm_atomic_host_allocations:
        ss << "usm_atomic_host_allocations";
        break;
    case aspect::usm_atomic_shared_allocations:
        ss << "usm_atomic_shared_allocations";
        break;
    case aspect::host_debuggable:
        ss << "host_debuggable";
        break;
    default:
        throw std::runtime_error("Unsupported aspect type");
    }
    return ss.str();
}

/*!
 * Transforms string to sycl::aspect.
 */
aspect DPCTL_StrToAspectType(const std::string &aspectTyStr)
{
    aspect aspectTy;
    if (aspectTyStr == "cpu") {
        aspectTy = aspect::cpu;
    }
#if __SYCL_COMPILER_VERSION < __SYCL_COMPILER_2023_SWITCHOVER
    else if (aspectTyStr == "host") {
        aspectTy = aspect::host;
    }
#endif
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
    else if (aspectTyStr == "atomic64") {
        aspectTy = aspect::atomic64;
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
    else if (aspectTyStr == "usm_system_allocations") {
        aspectTy = aspect::usm_system_allocations;
    }
    else if (aspectTyStr == "usm_atomic_host_allocations") {
        aspectTy = aspect::usm_atomic_host_allocations;
    }
    else if (aspectTyStr == "usm_atomic_shared_allocations") {
        aspectTy = aspect::usm_atomic_shared_allocations;
    }
    else if (aspectTyStr == "host_debuggable") {
        aspectTy = aspect::host_debuggable;
    }
    else {
        // \todo handle the error
        throw std::runtime_error("Unsupported aspect type");
    }
    return aspectTy;
}

aspect DPCTL_DPCTLAspectTypeToSyclAspect(DPCTLSyclAspectType AspectTy)
{
    switch (AspectTy) {
#if __SYCL_COMPILER_VERSION < __SYCL_COMPILER_2023_SWITCHOVER
    case DPCTLSyclAspectType::host:
        return aspect::host;
#endif
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
    case DPCTLSyclAspectType::atomic64:
        return aspect::atomic64;
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
    case DPCTLSyclAspectType::usm_system_allocations:
        return aspect::usm_system_allocations;
    case DPCTLSyclAspectType::usm_atomic_host_allocations:
        return aspect::usm_atomic_host_allocations;
    case DPCTLSyclAspectType::usm_atomic_shared_allocations:
        return aspect::usm_atomic_shared_allocations;
    case DPCTLSyclAspectType::host_debuggable:
        return aspect::host_debuggable;
    default:
        throw std::runtime_error("Unsupported aspect type");
    }
}

DPCTLSyclAspectType DPCTL_SyclAspectToDPCTLAspectType(aspect Aspect)
{
    switch (Aspect) {
#if __SYCL_COMPILER_VERSION < __SYCL_COMPILER_2023_SWITCHOVER
    case aspect::host:
        return DPCTLSyclAspectType::host;
#endif
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
    case aspect::atomic64:
        return DPCTLSyclAspectType::atomic64;
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
    case aspect::usm_system_allocations:
        return DPCTLSyclAspectType::usm_system_allocations;
    case aspect::usm_atomic_host_allocations:
        return DPCTLSyclAspectType::usm_atomic_host_allocations;
    case aspect::usm_atomic_shared_allocations:
        return DPCTLSyclAspectType::usm_atomic_shared_allocations;
    case aspect::host_debuggable:
        return DPCTLSyclAspectType::host_debuggable;
    default:
        throw std::runtime_error("Unsupported aspect type");
    }
}

info::partition_affinity_domain DPCTL_DPCTLPartitionAffinityDomainTypeToSycl(
    DPCTLPartitionAffinityDomainType PartitionAffinityDomainTy)
{
    switch (PartitionAffinityDomainTy) {
    case DPCTLPartitionAffinityDomainType::not_applicable:
        return info::partition_affinity_domain::not_applicable;
    case DPCTLPartitionAffinityDomainType::numa:
        return info::partition_affinity_domain::numa;
    case DPCTLPartitionAffinityDomainType::L4_cache:
        return info::partition_affinity_domain::L4_cache;
    case DPCTLPartitionAffinityDomainType::L3_cache:
        return info::partition_affinity_domain::L3_cache;
    case DPCTLPartitionAffinityDomainType::L2_cache:
        return info::partition_affinity_domain::L2_cache;
    case DPCTLPartitionAffinityDomainType::L1_cache:
        return info::partition_affinity_domain::L1_cache;
    case DPCTLPartitionAffinityDomainType::next_partitionable:
        return info::partition_affinity_domain::next_partitionable;
    default:
        throw std::runtime_error("Unsupported partition_affinity_domain type");
    }
}

DPCTLPartitionAffinityDomainType DPCTL_SyclPartitionAffinityDomainToDPCTLType(
    sycl::info::partition_affinity_domain PartitionAffinityDomain)
{
    switch (PartitionAffinityDomain) {
    case info::partition_affinity_domain::not_applicable:
        return DPCTLPartitionAffinityDomainType::not_applicable;
    case info::partition_affinity_domain::numa:
        return DPCTLPartitionAffinityDomainType::numa;
    case info::partition_affinity_domain::L4_cache:
        return DPCTLPartitionAffinityDomainType::L4_cache;
    case info::partition_affinity_domain::L3_cache:
        return DPCTLPartitionAffinityDomainType::L3_cache;
    case info::partition_affinity_domain::L2_cache:
        return DPCTLPartitionAffinityDomainType::L2_cache;
    case info::partition_affinity_domain::L1_cache:
        return DPCTLPartitionAffinityDomainType::L1_cache;
    case info::partition_affinity_domain::next_partitionable:
        return DPCTLPartitionAffinityDomainType::next_partitionable;
    default:
        throw std::runtime_error("Unsupported partition_affinity_domain type");
    }
}

int64_t DPCTL_GetRelativeDeviceId(const device &Device)
{
    auto relid = -1;
    auto p = Device.get_platform();
    auto be = p.get_backend();
    auto dt = Device.get_info<sycl::info::device::device_type>();
    auto dev_vec = device::get_devices(dt);
    int64_t id = 0;
    for (const auto &d_i : dev_vec) {
        if (Device == d_i) {
            relid = id;
            break;
        }
        if (d_i.get_platform().get_backend() == be)
            ++id;
    }
    return relid;
}

std::string DPCTL_GetDeviceFilterString(const device &Device)
{
    std::stringstream ss;
    static constexpr const char *filter_string_separator = ":";

    auto be = Device.get_platform().get_backend();

    switch (be) {
    case backend::ext_oneapi_level_zero:
        ss << "level_zero";
        break;
    case backend::ext_oneapi_cuda:
        ss << "cuda";
        break;
    case backend::opencl:
        ss << "opencl";
        break;
#if __SYCL_COMPILER_VERSION < __SYCL_COMPILER_2023_SWITCHOVER
    case backend::host:
        ss << "host";
        break;
#endif
    default:
        ss << "unknown";
    };

    ss << filter_string_separator;
    ss << DPCTL_DeviceTypeToStr(Device.get_info<info::device::device_type>());
    ss << filter_string_separator;
    ss << DPCTL_GetRelativeDeviceId(Device);

    return ss.str();
}

DPCTLSyclEventStatusType
DPCTL_SyclEventStatusToDPCTLEventStatusType(info::event_command_status E)
{
    switch (E) {
    case info::event_command_status::submitted:
        return DPCTLSyclEventStatusType::DPCTL_SUBMITTED;
    case info::event_command_status::running:
        return DPCTLSyclEventStatusType::DPCTL_RUNNING;
    case info::event_command_status::complete:
        return DPCTLSyclEventStatusType::DPCTL_COMPLETE;
    default:
        return DPCTLSyclEventStatusType::DPCTL_UNKNOWN_STATUS;
    }
}
