//===--- dpctl_sycl_device_interface.cpp - Implements C API for sycl::device =//
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
/// This file implements the data types and functions declared in
/// dpctl_sycl_device_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_device_interface.h"
#include "../helper/include/dpctl_utils_helper.h"
#include "Support/CBindingWrapping.h"
#include "dpctl_sycl_device_manager.h"
#include <CL/sycl.hpp> /* SYCL headers   */
#include <cstring>

using namespace cl::sycl;

namespace
{
// Create wrappers for C Binding types (see CBindingWrapping.h).
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(device, DPCTLSyclDeviceRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(device_selector, DPCTLSyclDeviceSelectorRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(platform, DPCTLSyclPlatformRef)

} /* end of anonymous namespace */

__dpctl_give DPCTLSyclDeviceRef
DPCTLDevice_Copy(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    auto Device = unwrap(DRef);
    if (!Device) {
        std::cerr << "Cannot copy DPCTLSyclDeviceRef as input is a nullptr\n";
        return nullptr;
    }
    try {
        auto CopiedDevice = new device(*Device);
        return wrap(CopiedDevice);
    } catch (std::bad_alloc const &ba) {
        // \todo log error
        std::cerr << ba.what() << '\n';
        return nullptr;
    }
}

__dpctl_give DPCTLSyclDeviceRef DPCTLDevice_Create()
{
    try {
        auto Device = new device();
        return wrap(Device);
    } catch (std::bad_alloc const &ba) {
        // \todo log error
        std::cerr << ba.what() << '\n';
        return nullptr;
    } catch (runtime_error const &re) {
        // \todo log error
        std::cerr << re.what() << '\n';
        return nullptr;
    }
}

__dpctl_give DPCTLSyclDeviceRef DPCTLDevice_CreateFromSelector(
    __dpctl_keep const DPCTLSyclDeviceSelectorRef DSRef)
{
    auto Selector = unwrap(DSRef);
    if (!Selector)
        // \todo : Log error
        return nullptr;
    try {
        auto Device = new device(*Selector);
        return wrap(Device);
    } catch (std::bad_alloc const &ba) {
        // \todo log error
        std::cerr << ba.what() << '\n';
        return nullptr;
    } catch (runtime_error const &re) {
        // \todo log error
        std::cerr << re.what() << '\n';
        return nullptr;
    }
}

void DPCTLDevice_Delete(__dpctl_take DPCTLSyclDeviceRef DRef)
{
    delete unwrap(DRef);
}

DPCTLSyclDeviceType
DPCTLDevice_GetDeviceType(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    DPCTLSyclDeviceType DTy = DPCTLSyclDeviceType::DPCTL_UNKNOWN_DEVICE;
    auto D = unwrap(DRef);
    if (D) {
        try {
            auto SyclDTy = D->get_info<info::device::device_type>();
            DTy = DPCTL_SyclDeviceTypeToDPCTLDeviceType(SyclDTy);
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    return DTy;
}

bool DPCTLDevice_IsAccelerator(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    auto D = unwrap(DRef);
    if (D) {
        return D->is_accelerator();
    }
    return false;
}

bool DPCTLDevice_IsCPU(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    auto D = unwrap(DRef);
    if (D) {
        return D->is_cpu();
    }
    return false;
}

bool DPCTLDevice_IsGPU(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    auto D = unwrap(DRef);
    if (D) {
        return D->is_gpu();
    }
    return false;
}

bool DPCTLDevice_IsHost(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    auto D = unwrap(DRef);
    if (D) {
        return D->is_host();
    }
    return false;
}

DPCTLSyclBackendType
DPCTLDevice_GetBackend(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    DPCTLSyclBackendType BTy = DPCTLSyclBackendType::DPCTL_UNKNOWN_BACKEND;
    auto D = unwrap(DRef);
    if (D) {
        BTy = DPCTL_SyclBackendToDPCTLBackendType(
            D->get_platform().get_backend());
    }
    return BTy;
}

uint32_t
DPCTLDevice_GetMaxComputeUnits(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    uint32_t nComputeUnits = 0;
    auto D = unwrap(DRef);
    if (D) {
        try {
            nComputeUnits = D->get_info<info::device::max_compute_units>();
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    return nComputeUnits;
}

uint32_t
DPCTLDevice_GetMaxWorkItemDims(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    uint32_t maxWorkItemDims = 0;
    auto D = unwrap(DRef);
    if (D) {
        try {
            maxWorkItemDims =
                D->get_info<info::device::max_work_item_dimensions>();
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    return maxWorkItemDims;
}

__dpctl_keep size_t *
DPCTLDevice_GetMaxWorkItemSizes(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    size_t *sizes = nullptr;
    auto D = unwrap(DRef);
    if (D) {
        try {
            auto id_sizes = D->get_info<info::device::max_work_item_sizes>();
            sizes = new size_t[3];
            for (auto i = 0ul; i < 3; ++i) {
                sizes[i] = id_sizes[i];
            }
        } catch (std::bad_alloc const &ba) {
            // \todo log error
            std::cerr << ba.what() << '\n';
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    return sizes;
}

size_t
DPCTLDevice_GetMaxWorkGroupSize(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    size_t max_wg_size = 0;
    auto D = unwrap(DRef);
    if (D) {
        try {
            max_wg_size = D->get_info<info::device::max_work_group_size>();
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    return max_wg_size;
}

uint32_t
DPCTLDevice_GetMaxNumSubGroups(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    size_t max_nsubgroups = 0;
    auto D = unwrap(DRef);
    if (D) {
        try {
            max_nsubgroups = D->get_info<info::device::max_num_sub_groups>();
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    return max_nsubgroups;
}

__dpctl_give DPCTLSyclPlatformRef
DPCTLDevice_GetPlatform(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    DPCTLSyclPlatformRef PRef = nullptr;
    auto D = unwrap(DRef);
    if (D) {
        try {
            PRef = wrap(new platform(D->get_platform()));
        } catch (std::bad_alloc const &ba) {
            std::cerr << ba.what() << '\n';
        }
    }
    return PRef;
}

__dpctl_give const char *
DPCTLDevice_GetName(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    char *cstr_name = nullptr;
    auto D = unwrap(DRef);
    if (D) {
        try {
            auto name = D->get_info<info::device::name>();
            auto cstr_len = name.length() + 1;
            cstr_name = new char[cstr_len];
#ifdef _WIN32
            strncpy_s(cstr_name, cstr_len, name.c_str(), cstr_len);
#else
            std::strncpy(cstr_name, name.c_str(), cstr_len);
#endif
        } catch (std::bad_alloc const &ba) {
            // \todo log error
            std::cerr << ba.what() << '\n';
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    return cstr_name;
}

__dpctl_give const char *
DPCTLDevice_GetVendorName(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    char *cstr_vendor = nullptr;
    auto D = unwrap(DRef);
    if (D) {
        try {
            auto vendor = D->get_info<info::device::vendor>();
            auto cstr_len = vendor.length() + 1;
            cstr_vendor = new char[cstr_len];
#ifdef _WIN32
            strncpy_s(cstr_vendor, cstr_len, vendor.c_str(), cstr_len);
#else
            std::strncpy(cstr_vendor, vendor.c_str(), cstr_len);
#endif
        } catch (std::bad_alloc const &ba) {
            // \todo log error
            std::cerr << ba.what() << '\n';
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    return cstr_vendor;
}

__dpctl_give const char *
DPCTLDevice_GetDriverInfo(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    char *cstr_driver = nullptr;
    auto D = unwrap(DRef);
    if (D) {
        try {
            auto driver = D->get_info<info::device::driver_version>();
            auto cstr_len = driver.length() + 1;
            cstr_driver = new char[cstr_len];
#ifdef _WIN32
            strncpy_s(cstr_driver, cstr_len, driver.c_str(), cstr_len);
#else
            std::strncpy(cstr_driver, driver.c_str(), cstr_len);
#endif
        } catch (std::bad_alloc const &ba) {
            // \todo log error
            std::cerr << ba.what() << '\n';
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    return cstr_driver;
}

bool DPCTLDevice_IsHostUnifiedMemory(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    bool ret = false;
    auto D = unwrap(DRef);
    if (D) {
        try {
            ret = D->get_info<info::device::host_unified_memory>();
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    return ret;
}

bool DPCTLDevice_AreEq(__dpctl_keep const DPCTLSyclDeviceRef DRef1,
                       __dpctl_keep const DPCTLSyclDeviceRef DRef2)
{
    // Note: DPCPP does not yet support device equality of the form:
    // *unwrap(DevRef1) == *unwrap(DevRef2). Till DPCPP is fixed we use the
    // custom equality checker implemented inside DPCTLDeviceMgr.
    return DPCTLDeviceMgr_AreEq(DRef1, DRef2);
}

bool DPCTLDevice_HasAspect(__dpctl_keep const DPCTLSyclDeviceRef DRef,
                           DPCTLSyclAspectType AT)
{
    bool hasAspect = false;
    auto D = unwrap(DRef);
    if (D) {
        try {
            hasAspect = D->has(DPCTL_DPCTLAspectTypeToSyclAspect(AT));
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    return hasAspect;
}

bool DPCTLDevice_GetSubGroupIndependentForwardProgress(
    __dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    bool SubGroupProgress = false;
    auto D = unwrap(DRef);
    if (D) {
        try {
            SubGroupProgress = D->get_info<
                info::device::sub_group_independent_forward_progress>();
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    return SubGroupProgress;
}

uint32_t DPCTLDevice_GetPreferredVectorWidthChar(
    __dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    size_t vector_width_char = 0;
    auto D = unwrap(DRef);
    if (D) {
        try {
            vector_width_char =
                D->get_info<info::device::preferred_vector_width_char>();
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    return vector_width_char;
}

uint32_t DPCTLDevice_GetPreferredVectorWidthShort(
    __dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    size_t vector_width_short = 0;
    auto D = unwrap(DRef);
    if (D) {
        try {
            vector_width_short =
                D->get_info<info::device::preferred_vector_width_short>();
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    return vector_width_short;
}

uint32_t DPCTLDevice_GetPreferredVectorWidthInt(
    __dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    size_t vector_width_int = 0;
    auto D = unwrap(DRef);
    if (D) {
        try {
            vector_width_int =
                D->get_info<info::device::preferred_vector_width_int>();
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    return vector_width_int;
}

uint32_t DPCTLDevice_GetPreferredVectorWidthLong(
    __dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    size_t vector_width_long = 0;
    auto D = unwrap(DRef);
    if (D) {
        try {
            vector_width_long =
                D->get_info<info::device::preferred_vector_width_long>();
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    return vector_width_long;
}

uint32_t DPCTLDevice_GetPreferredVectorWidthFloat(
    __dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    size_t vector_width_float = 0;
    auto D = unwrap(DRef);
    if (D) {
        try {
            vector_width_float =
                D->get_info<info::device::preferred_vector_width_float>();
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    return vector_width_float;
}

uint32_t DPCTLDevice_GetPreferredVectorWidthDouble(
    __dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    size_t vector_width_double = 0;
    auto D = unwrap(DRef);
    if (D) {
        try {
            vector_width_double =
                D->get_info<info::device::preferred_vector_width_double>();
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    return vector_width_double;
}

uint32_t DPCTLDevice_GetPreferredVectorWidthHalf(
    __dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    size_t vector_width_half = 0;
    auto D = unwrap(DRef);
    if (D) {
        try {
            vector_width_half =
                D->get_info<info::device::preferred_vector_width_half>();
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    return vector_width_half;
}

__dpctl_give DPCTLDeviceVectorRef
DPCTLDevice_CreateSubDevicesEqually(__dpctl_keep const DPCTLSyclDeviceRef DRef,
                                    size_t count)
{
    vector_class<DPCTLSyclDeviceRef> *Devices = nullptr;
    auto D = unwrap(DRef);
    if (D) {
        try {
            auto subDevices = D->create_sub_devices<
                info::partition_property::partition_equally>(count);
            Devices = new vector_class<DPCTLSyclDeviceRef>();
            for (const auto &sd : subDevices) {
                Devices->emplace_back(wrap(new device(sd)));
            }
        } catch (std::bad_alloc const &ba) {
            std::cerr << ba.what() << '\n';
            return nullptr;
        } catch (feature_not_supported const &fnse) {
            std::cerr << fnse.what() << '\n';
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    return wrap(Devices);
}

__dpctl_give DPCTLDeviceVectorRef
DPCTLDevice_CreateSubDevicesByCounts(__dpctl_keep const DPCTLSyclDeviceRef DRef,
                                     __dpctl_keep size_t *counts,
                                     size_t ncounts)
{
    vector_class<DPCTLSyclDeviceRef> *Devices = nullptr;
    std::vector<size_t> vcounts;
    vcounts.assign(counts, counts + ncounts);
    auto D = unwrap(DRef);
    if (D) {
        try {
            auto subDevices = D->create_sub_devices<
                info::partition_property::partition_by_counts>(vcounts);
            Devices = new vector_class<DPCTLSyclDeviceRef>();
            for (auto &sd : subDevices) {
                Devices->emplace_back(wrap(new device(sd)));
            }
        } catch (std::bad_alloc const &ba) {
            std::cerr << ba.what() << '\n';
            return nullptr;
        } catch (feature_not_supported const &fnse) {
            std::cerr << fnse.what() << '\n';
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    return wrap(Devices);
}

__dpctl_give DPCTLDeviceVectorRef DPCTLDevice_CreateSubDevicesByAffinity(
    __dpctl_keep const DPCTLSyclDeviceRef DRef,
    DPCTLPartitionAffinityDomainType PartitionAffinityDomainTy)
{
    vector_class<DPCTLSyclDeviceRef> *Devices = nullptr;
    auto D = unwrap(DRef);
    if (D) {
        try {
            auto domain = DPCTL_DPCTLPartitionAffinityDomainTypeToSycl(
                PartitionAffinityDomainTy);
            auto subDevices = D->create_sub_devices<
                info::partition_property::partition_by_affinity_domain>(domain);
            Devices = new vector_class<DPCTLSyclDeviceRef>();
            for (auto &sd : subDevices) {
                Devices->emplace_back(wrap(new device(sd)));
            }
        } catch (std::bad_alloc const &ba) {
            std::cerr << ba.what() << '\n';
            return nullptr;
        } catch (feature_not_supported const &fnse) {
            std::cerr << fnse.what() << '\n';
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    return wrap(Devices);
}
