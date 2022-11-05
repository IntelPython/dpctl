//===--- dpctl_sycl_device_interface.cpp - Implements C API for sycl::device =//
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
/// This file implements the data types and functions declared in
/// dpctl_sycl_device_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_device_interface.h"
#include "Config/dpctl_config.h"
#include "dpctl_device_selection.hpp"
#include "dpctl_error_handlers.h"
#include "dpctl_string_utils.hpp"
#include "dpctl_sycl_device_manager.h"
#include "dpctl_sycl_type_casters.hpp"
#include "dpctl_utils_helper.h"
#include <CL/sycl.hpp> /* SYCL headers   */
#include <algorithm>
#include <cstring>
#include <vector>

using namespace sycl;

namespace
{

using namespace dpctl::syclinterface;

device *new_device_from_selector(const dpctl_device_selector *sel)
{
#if __SYCL_COMPILER_VERSION >= __SYCL_COMPILER_2023_SWITCHOVER
    return new device(
        [=](const device &d) -> int { return sel->operator()(d); });
#else
    return new device(*sel);
#endif
}

template <int dim>
__dpctl_keep size_t *
DPCTLDevice__GetMaxWorkItemSizes(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    size_t *sizes = nullptr;
    auto D = unwrap<device>(DRef);
    if (D) {
        try {
#if __SYCL_COMPILER_VERSION >= __SYCL_COMPILER_MAX_WORK_ITEM_SIZE_THRESHOLD
            auto id_sizes =
                D->get_info<info::device::max_work_item_sizes<dim>>();
#else
            auto id_sizes = D->get_info<info::device::max_work_item_sizes>();
#endif
            sizes = new size_t[dim];
            for (auto i = 0ul; i < dim; ++i) {
                sizes[i] = id_sizes[i];
            }
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    return sizes;
}

} /* end of anonymous namespace */

__dpctl_give DPCTLSyclDeviceRef
DPCTLDevice_Copy(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    auto Device = unwrap<device>(DRef);
    if (!Device) {
        error_handler("Cannot copy DPCTLSyclDeviceRef as input is a nullptr",
                      __FILE__, __func__, __LINE__);
        return nullptr;
    }
    try {
        auto CopiedDevice = new device(*Device);
        return wrap<device>(CopiedDevice);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

__dpctl_give DPCTLSyclDeviceRef DPCTLDevice_Create()
{
    try {
        auto Device = new device();
        return wrap<device>(Device);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

__dpctl_give DPCTLSyclDeviceRef DPCTLDevice_CreateFromSelector(
    __dpctl_keep const DPCTLSyclDeviceSelectorRef DSRef)
{
    auto Selector = unwrap<dpctl_device_selector>(DSRef);
    if (!Selector) {
        error_handler("Cannot difine device selector for DPCTLSyclDeviceRef "
                      "as input is a nullptr.",
                      __FILE__, __func__, __LINE__);
        return nullptr;
    }
    try {
        auto Device = new_device_from_selector(Selector);
        return wrap<device>(Device);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

void DPCTLDevice_Delete(__dpctl_take DPCTLSyclDeviceRef DRef)
{
    delete unwrap<device>(DRef);
}

DPCTLSyclDeviceType
DPCTLDevice_GetDeviceType(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    DPCTLSyclDeviceType DTy = DPCTLSyclDeviceType::DPCTL_UNKNOWN_DEVICE;
    auto D = unwrap<device>(DRef);
    if (D) {
        try {
            auto SyclDTy = D->get_info<info::device::device_type>();
            DTy = DPCTL_SyclDeviceTypeToDPCTLDeviceType(SyclDTy);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    return DTy;
}

bool DPCTLDevice_IsAccelerator(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    auto D = unwrap<device>(DRef);
    if (D) {
        return D->is_accelerator();
    }
    return false;
}

bool DPCTLDevice_IsCPU(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    auto D = unwrap<device>(DRef);
    if (D) {
        return D->is_cpu();
    }
    return false;
}

bool DPCTLDevice_IsGPU(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    auto D = unwrap<device>(DRef);
    if (D) {
        return D->is_gpu();
    }
    return false;
}

bool DPCTLDevice_IsHost(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    auto D = unwrap<device>(DRef);
    if (D) {
#if __SYCL_COMPILER_VERSION >= __SYCL_COMPILER_2023_SWITCHOVER
        return false;
#else
        return D->is_host();
#endif
    }
    return false;
}

DPCTLSyclBackendType
DPCTLDevice_GetBackend(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    DPCTLSyclBackendType BTy = DPCTLSyclBackendType::DPCTL_UNKNOWN_BACKEND;
    auto D = unwrap<device>(DRef);
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
    auto D = unwrap<device>(DRef);
    if (D) {
        try {
            nComputeUnits = D->get_info<info::device::max_compute_units>();
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    return nComputeUnits;
}

uint64_t
DPCTLDevice_GetGlobalMemSize(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    uint64_t GlobalMemSize = 0;
    auto D = unwrap<device>(DRef);
    if (D) {
        try {
            GlobalMemSize = D->get_info<info::device::global_mem_size>();
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    return GlobalMemSize;
}

uint64_t DPCTLDevice_GetLocalMemSize(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    uint64_t LocalMemSize = 0;
    auto D = unwrap<device>(DRef);
    if (D) {
        try {
            LocalMemSize = D->get_info<info::device::local_mem_size>();
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    return LocalMemSize;
}

uint32_t
DPCTLDevice_GetMaxWorkItemDims(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    uint32_t maxWorkItemDims = 0;
    auto D = unwrap<device>(DRef);
    if (D) {
        try {
            maxWorkItemDims =
                D->get_info<info::device::max_work_item_dimensions>();
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    return maxWorkItemDims;
}

__dpctl_keep size_t *
DPCTLDevice_GetMaxWorkItemSizes1d(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    return DPCTLDevice__GetMaxWorkItemSizes<1>(DRef);
}

__dpctl_keep size_t *
DPCTLDevice_GetMaxWorkItemSizes2d(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    return DPCTLDevice__GetMaxWorkItemSizes<2>(DRef);
}

__dpctl_keep size_t *
DPCTLDevice_GetMaxWorkItemSizes3d(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    return DPCTLDevice__GetMaxWorkItemSizes<3>(DRef);
}

__dpctl_keep size_t *
DPCTLDevice_GetMaxWorkItemSizes(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    return DPCTLDevice__GetMaxWorkItemSizes<3>(DRef);
}

size_t
DPCTLDevice_GetMaxWorkGroupSize(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    size_t max_wg_size = 0;
    auto D = unwrap<device>(DRef);
    if (D) {
        try {
            max_wg_size = D->get_info<info::device::max_work_group_size>();
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    return max_wg_size;
}

uint32_t
DPCTLDevice_GetMaxNumSubGroups(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    size_t max_nsubgroups = 0;
    auto D = unwrap<device>(DRef);
    if (D) {
        try {
            max_nsubgroups = D->get_info<info::device::max_num_sub_groups>();
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    return max_nsubgroups;
}

__dpctl_give DPCTLSyclPlatformRef
DPCTLDevice_GetPlatform(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    DPCTLSyclPlatformRef PRef = nullptr;
    auto D = unwrap<device>(DRef);
    if (D) {
        try {
            PRef = wrap<platform>(new platform(D->get_platform()));
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    return PRef;
}

__dpctl_give const char *
DPCTLDevice_GetName(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    const char *cstr_name = nullptr;
    auto D = unwrap<device>(DRef);
    if (D) {
        try {
            auto name = D->get_info<info::device::name>();
            cstr_name = dpctl::helper::cstring_from_string(name);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    return cstr_name;
}

__dpctl_give const char *
DPCTLDevice_GetVendor(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    const char *cstr_vendor = nullptr;
    auto D = unwrap<device>(DRef);
    if (D) {
        try {
            auto vendor = D->get_info<info::device::vendor>();
            cstr_vendor = dpctl::helper::cstring_from_string(vendor);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    return cstr_vendor;
}

__dpctl_give const char *
DPCTLDevice_GetDriverVersion(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    const char *cstr_driver = nullptr;
    auto D = unwrap<device>(DRef);
    if (D) {
        try {
            auto driver = D->get_info<info::device::driver_version>();
            cstr_driver = dpctl::helper::cstring_from_string(driver);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    return cstr_driver;
}

bool DPCTLDevice_AreEq(__dpctl_keep const DPCTLSyclDeviceRef DRef1,
                       __dpctl_keep const DPCTLSyclDeviceRef DRef2)
{
    auto D1 = unwrap<device>(DRef1);
    auto D2 = unwrap<device>(DRef2);
    if (D1 && D2)
        return *D1 == *D2;
    else
        return false;
}

bool DPCTLDevice_HasAspect(__dpctl_keep const DPCTLSyclDeviceRef DRef,
                           DPCTLSyclAspectType AT)
{
    bool hasAspect = false;
    auto D = unwrap<device>(DRef);
    if (D) {
        try {
            hasAspect = D->has(DPCTL_DPCTLAspectTypeToSyclAspect(AT));
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    return hasAspect;
}

#define declmethod(FUNC, NAME, TYPE)                                           \
    TYPE DPCTLDevice_##FUNC(__dpctl_keep const DPCTLSyclDeviceRef DRef)        \
    {                                                                          \
        TYPE result = 0;                                                       \
        auto D = unwrap<device>(DRef);                                         \
        if (D) {                                                               \
            try {                                                              \
                result = D->get_info<info::device::NAME>();                    \
            } catch (std::exception const &e) {                                \
                error_handler(e, __FILE__, __func__, __LINE__);                \
            }                                                                  \
        }                                                                      \
        return result;                                                         \
    }
declmethod(GetMaxReadImageArgs, max_read_image_args, uint32_t);
declmethod(GetMaxWriteImageArgs, max_write_image_args, uint32_t);
declmethod(GetImage2dMaxWidth, image2d_max_width, size_t);
declmethod(GetImage2dMaxHeight, image2d_max_height, size_t);
declmethod(GetImage3dMaxWidth, image3d_max_width, size_t);
declmethod(GetImage3dMaxHeight, image3d_max_height, size_t);
declmethod(GetImage3dMaxDepth, image3d_max_depth, size_t);
#undef declmethod

bool DPCTLDevice_GetSubGroupIndependentForwardProgress(
    __dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    bool SubGroupProgress = false;
    auto D = unwrap<device>(DRef);
    if (D) {
        try {
            SubGroupProgress = D->get_info<
                info::device::sub_group_independent_forward_progress>();
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    return SubGroupProgress;
}

uint32_t DPCTLDevice_GetPreferredVectorWidthChar(
    __dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    size_t vector_width_char = 0;
    auto D = unwrap<device>(DRef);
    if (D) {
        try {
            vector_width_char =
                D->get_info<info::device::preferred_vector_width_char>();
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    return vector_width_char;
}

uint32_t DPCTLDevice_GetPreferredVectorWidthShort(
    __dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    size_t vector_width_short = 0;
    auto D = unwrap<device>(DRef);
    if (D) {
        try {
            vector_width_short =
                D->get_info<info::device::preferred_vector_width_short>();
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    return vector_width_short;
}

uint32_t DPCTLDevice_GetPreferredVectorWidthInt(
    __dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    size_t vector_width_int = 0;
    auto D = unwrap<device>(DRef);
    if (D) {
        try {
            vector_width_int =
                D->get_info<info::device::preferred_vector_width_int>();
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    return vector_width_int;
}

uint32_t DPCTLDevice_GetPreferredVectorWidthLong(
    __dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    size_t vector_width_long = 0;
    auto D = unwrap<device>(DRef);
    if (D) {
        try {
            vector_width_long =
                D->get_info<info::device::preferred_vector_width_long>();
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    return vector_width_long;
}

uint32_t DPCTLDevice_GetPreferredVectorWidthFloat(
    __dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    size_t vector_width_float = 0;
    auto D = unwrap<device>(DRef);
    if (D) {
        try {
            vector_width_float =
                D->get_info<info::device::preferred_vector_width_float>();
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    return vector_width_float;
}

uint32_t DPCTLDevice_GetPreferredVectorWidthDouble(
    __dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    size_t vector_width_double = 0;
    auto D = unwrap<device>(DRef);
    if (D) {
        try {
            vector_width_double =
                D->get_info<info::device::preferred_vector_width_double>();
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    return vector_width_double;
}

uint32_t DPCTLDevice_GetPreferredVectorWidthHalf(
    __dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    size_t vector_width_half = 0;
    auto D = unwrap<device>(DRef);
    if (D) {
        try {
            vector_width_half =
                D->get_info<info::device::preferred_vector_width_half>();
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    return vector_width_half;
}

__dpctl_give DPCTLSyclDeviceRef
DPCTLDevice_GetParentDevice(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    auto D = unwrap<device>(DRef);
    if (D) {
        try {
            auto parent_D = D->get_info<info::device::parent_device>();
            return wrap<device>(new device(parent_D));
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
            return nullptr;
        }
    }
    else
        return nullptr;
}

__dpctl_give DPCTLDeviceVectorRef
DPCTLDevice_CreateSubDevicesEqually(__dpctl_keep const DPCTLSyclDeviceRef DRef,
                                    size_t count)
{
    using vecTy = std::vector<DPCTLSyclDeviceRef>;
    vecTy *Devices = nullptr;
    if (DRef) {
        if (count == 0) {
            error_handler("Cannot create sub-devices with zero compute units",
                          __FILE__, __func__, __LINE__);
            return nullptr;
        }
        auto D = unwrap<device>(DRef);
        try {
            auto subDevices = D->create_sub_devices<
                info::partition_property::partition_equally>(count);
            Devices = new vecTy();
            for (const auto &sd : subDevices) {
                Devices->emplace_back(wrap<device>(new device(sd)));
            }
        } catch (std::exception const &e) {
            delete Devices;
            error_handler(e, __FILE__, __func__, __LINE__);
            return nullptr;
        }
    }
    return wrap<vecTy>(Devices);
}

__dpctl_give DPCTLDeviceVectorRef
DPCTLDevice_CreateSubDevicesByCounts(__dpctl_keep const DPCTLSyclDeviceRef DRef,
                                     __dpctl_keep size_t *counts,
                                     size_t ncounts)
{
    using vecTy = std::vector<DPCTLSyclDeviceRef>;
    vecTy *Devices = nullptr;
    std::vector<size_t> vcounts(ncounts);
    vcounts.assign(counts, counts + ncounts);
    size_t min_elem = *std::min_element(vcounts.begin(), vcounts.end());
    if (min_elem == 0) {
        error_handler("Cannot create sub-devices with zero compute units",
                      __FILE__, __func__, __LINE__);
        return nullptr;
    }
    if (DRef) {
        auto D = unwrap<device>(DRef);
        std::vector<std::remove_pointer<decltype(D)>::type> subDevices;
        try {
            subDevices = D->create_sub_devices<
                info::partition_property::partition_by_counts>(vcounts);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
            return nullptr;
        }
        try {
            Devices = new vecTy();
            for (const auto &sd : subDevices) {
                Devices->emplace_back(wrap<device>(new device(sd)));
            }
        } catch (std::exception const &e) {
            delete Devices;
            error_handler(e, __FILE__, __func__, __LINE__);
            return nullptr;
        }
    }
    return wrap<vecTy>(Devices);
}

__dpctl_give DPCTLDeviceVectorRef DPCTLDevice_CreateSubDevicesByAffinity(
    __dpctl_keep const DPCTLSyclDeviceRef DRef,
    DPCTLPartitionAffinityDomainType PartitionAffinityDomainTy)
{
    using vecTy = std::vector<DPCTLSyclDeviceRef>;
    vecTy *Devices = nullptr;
    auto D = unwrap<device>(DRef);
    if (D) {
        try {
            auto domain = DPCTL_DPCTLPartitionAffinityDomainTypeToSycl(
                PartitionAffinityDomainTy);
            auto subDevices = D->create_sub_devices<
                info::partition_property::partition_by_affinity_domain>(domain);
            Devices = new vecTy();
            for (const auto &sd : subDevices) {
                Devices->emplace_back(wrap<device>(new device(sd)));
            }
        } catch (std::exception const &e) {
            delete Devices;
            error_handler(e, __FILE__, __func__, __LINE__);
            return nullptr;
        }
    }
    return wrap<vecTy>(Devices);
}

size_t DPCTLDevice_Hash(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    if (DRef) {
        auto D = unwrap<device>(DRef);
        std::hash<device> hash_fn;
        return hash_fn(*D);
    }
    else {
        error_handler("Argument DRef is null", __FILE__, __func__, __LINE__);
        return 0;
    }
}

size_t DPCTLDevice_GetProfilingTimerResolution(
    __dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    if (DRef) {
        auto D = unwrap<device>(DRef);
        return D->get_info<info::device::profiling_timer_resolution>();
    }
    else {
        error_handler("Argument DRef is null", __FILE__, __func__, __LINE__);
        return 0;
    }
}

uint32_t DPCTLDevice_GetGlobalMemCacheLineSize(
    __dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    if (DRef) {
        auto D = unwrap<device>(DRef);
        return D->get_info<info::device::global_mem_cache_line_size>();
    }
    else {
        error_handler("Argument DRef is null", __FILE__, __func__, __LINE__);
        return 0;
    }
}

uint64_t
DPCTLDevice_GetGlobalMemCacheSize(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    if (DRef) {
        auto D = unwrap<device>(DRef);
        return D->get_info<info::device::global_mem_cache_size>();
    }
    else {
        error_handler("Argument DRef is null", __FILE__, __func__, __LINE__);
        return 0;
    }
}

DPCTLGlobalMemCacheType
DPCTLDevice_GetGlobalMemCacheType(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    if (DRef) {
        auto D = unwrap<device>(DRef);
        auto mem_type = D->get_info<info::device::global_mem_cache_type>();
        switch (mem_type) {
        case info::global_mem_cache_type::none:
            return DPCTL_MEM_CACHE_TYPE_NONE;
        case info::global_mem_cache_type::read_only:
            return DPCTL_MEM_CACHE_TYPE_READ_ONLY;
        case info::global_mem_cache_type::read_write:
            return DPCTL_MEM_CACHE_TYPE_READ_WRITE;
        }
        // If execution reaches here unrecognized mem_type was returned. Check
        // values in the enumeration `info::global_mem_cache_type` in SYCL specs
        assert(false);
        return DPCTL_MEM_CACHE_TYPE_INDETERMINATE;
    }
    else {
        error_handler("Argument DRef is null", __FILE__, __func__, __LINE__);
        return DPCTL_MEM_CACHE_TYPE_INDETERMINATE;
    }
}
