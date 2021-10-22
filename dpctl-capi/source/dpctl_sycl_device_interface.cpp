//===--- dpctl_sycl_device_interface.cpp - Implements C API for sycl::device =//
//
//                      Data Parallel Control (dpctl)
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
#include "../helper/include/dpctl_error_handlers.h"
#include "../helper/include/dpctl_string_utils.hpp"
#include "../helper/include/dpctl_utils_helper.h"
#include "Support/CBindingWrapping.h"
#include "dpctl_sycl_device_manager.h"
#include <CL/sycl.hpp> /* SYCL headers   */
#include <algorithm>
#include <cstring>
#include <vector>

using namespace cl::sycl;

namespace
{
// Create wrappers for C Binding types (see CBindingWrapping.h).
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(device, DpctlSyclDeviceRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(device_selector, DpctlSyclDeviceSelectorRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(platform, DpctlSyclPlatformRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(std::vector<DpctlSyclDeviceRef>,
                                   DPCTLDeviceVectorRef)
} /* end of anonymous namespace */

__dpctl_give DpctlSyclDeviceRef
dpctl_device_copy(__dpctl_keep const DpctlSyclDeviceRef DRef,
                  __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;

    auto Device = unwrap(DRef);

    if (!Device) {
        handler(-1, "Cannot copy DpctlSyclDeviceRef as input is a nullptr.",
                __FILE__, __func__, __LINE__);
        return nullptr;
    }
    try {
        auto CopiedDevice = new device(*Device);
        return wrap(CopiedDevice);
    } catch (std::bad_alloc const &ba) {
        handler(-1, ba.what(), __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

__dpctl_give DpctlSyclDeviceRef
dpctl_device_create(__dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
    try {
        auto Device = new device();
        return wrap(Device);
    } catch (std::bad_alloc const &ba) {
        handler(-1, ba.what(), __FILE__, __func__, __LINE__);
        return nullptr;
    } catch (const sycl::exception &e) {
        handler(e.get_cl_code(), e.what(), __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

__dpctl_give DpctlSyclDeviceRef dpctl_device_create_from_selector(
    __dpctl_keep const DpctlSyclDeviceSelectorRef DSRef,
    __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
    auto Selector = unwrap(DSRef);
    if (!Selector)
        // \todo : Log error
        return nullptr;
    try {
        auto Device = new device(*Selector);
        return wrap(Device);
    } catch (std::bad_alloc const &ba) {
        handler(-1, ba.what(), __FILE__, __func__, __LINE__);
        return nullptr;
    } catch (runtime_error const &re) {
        // \todo log error
        std::cerr << re.what() << '\n';
        return nullptr;
    }
}

void dpctl_device_delete(__dpctl_take DpctlSyclDeviceRef DRef,
                         __dpctl_keep const DpctlExecState /**/)
{
    delete unwrap(DRef);
}

DPCTLSyclDeviceType
dpctl_device_get_device_type(__dpctl_keep const DpctlSyclDeviceRef DRef,
                             __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
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

bool dpctl_device_is_accelerator(__dpctl_keep const DpctlSyclDeviceRef DRef,
                                 __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
    auto D = unwrap(DRef);
    if (D) {
        return D->is_accelerator();
    }
    return false;
}

bool dpctl_device_isCPU(__dpctl_keep const DpctlSyclDeviceRef DRef,
                        __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
    auto D = unwrap(DRef);
    if (D) {
        return D->is_cpu();
    }
    return false;
}

bool dpctl_device_isGPU(__dpctl_keep const DpctlSyclDeviceRef DRef,
                        __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
    auto D = unwrap(DRef);
    if (D) {
        return D->is_gpu();
    }
    return false;
}

bool dpctl_device_is_host(__dpctl_keep const DpctlSyclDeviceRef DRef,
                          __dpctl_keep const DpctlExecState ES)
{
    auto D = unwrap(DRef);
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
    if (D) {
        return D->is_host();
    }
    return false;
}

DPCTLSyclBackendType
dpctl_device_get_backend(__dpctl_keep const DpctlSyclDeviceRef DRef,
                         __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
    DPCTLSyclBackendType BTy = DPCTLSyclBackendType::DPCTL_UNKNOWN_BACKEND;
    auto D = unwrap(DRef);
    if (D) {
        BTy = DPCTL_SyclBackendToDPCTLBackendType(
            D->get_platform().get_backend());
    }
    return BTy;
}

uint32_t
dpctl_device_get_max_compute_units(__dpctl_keep const DpctlSyclDeviceRef DRef,
                                   __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
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

uint64_t
dpctl_device_get_global_mem_size(__dpctl_keep const DpctlSyclDeviceRef DRef,
                                 __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
    uint64_t GlobalMemSize = 0;
    auto D = unwrap(DRef);
    if (D) {
        try {
            GlobalMemSize = D->get_info<info::device::global_mem_size>();
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    return GlobalMemSize;
}

uint64_t
dpctl_device_get_local_mem_size(__dpctl_keep const DpctlSyclDeviceRef DRef,
                                __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
    uint64_t LocalMemSize = 0;
    auto D = unwrap(DRef);
    if (D) {
        try {
            LocalMemSize = D->get_info<info::device::local_mem_size>();
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    return LocalMemSize;
}

uint32_t
dpctl_device_get_max_work_item_dims(__dpctl_keep const DpctlSyclDeviceRef DRef,
                                    __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
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
dpctl_device_get_max_work_item_sizes(__dpctl_keep const DpctlSyclDeviceRef DRef,
                                     __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
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
dpctl_device_get_max_work_group_size(__dpctl_keep const DpctlSyclDeviceRef DRef,
                                     __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
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
dpctl_device_get_max_num_sub_groups(__dpctl_keep const DpctlSyclDeviceRef DRef,
                                    __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
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

__dpctl_give DpctlSyclPlatformRef
dpctl_device_get_platform(__dpctl_keep const DpctlSyclDeviceRef DRef,
                          __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
    DpctlSyclPlatformRef PRef = nullptr;
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
dpctl_device_get_name(__dpctl_keep const DpctlSyclDeviceRef DRef,
                      __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
    const char *cstr_name = nullptr;
    auto D = unwrap(DRef);
    if (D) {
        try {
            auto name = D->get_info<info::device::name>();
            cstr_name = dpctl::helper::cstring_from_string(name);
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    return cstr_name;
}

__dpctl_give const char *
dpctl_device_get_vendor(__dpctl_keep const DpctlSyclDeviceRef DRef,
                        __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
    const char *cstr_vendor = nullptr;
    auto D = unwrap(DRef);
    if (D) {
        try {
            auto vendor = D->get_info<info::device::vendor>();
            cstr_vendor = dpctl::helper::cstring_from_string(vendor);
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    return cstr_vendor;
}

__dpctl_give const char *
dpctl_device_get_driver_version(__dpctl_keep const DpctlSyclDeviceRef DRef,
                                __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
    const char *cstr_driver = nullptr;
    auto D = unwrap(DRef);
    if (D) {
        try {
            auto driver = D->get_info<info::device::driver_version>();
            cstr_driver = dpctl::helper::cstring_from_string(driver);
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
        }
    }
    return cstr_driver;
}

bool dpctl_device_is_host_unified_memory(__dpctl_keep const DpctlSyclDeviceRef
                                             DRef,
                                         __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
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

bool dpctl_device_are_eq(__dpctl_keep const DpctlSyclDeviceRef DRef1,
                         __dpctl_keep const DpctlSyclDeviceRef DRef2,
                         __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
    auto D1 = unwrap(DRef1);
    auto D2 = unwrap(DRef2);
    if (D1 && D2)
        return *D1 == *D2;
    else
        return false;
}

bool dpctl_device_has_aspect(__dpctl_keep const DpctlSyclDeviceRef DRef,
                             DPCTLSyclAspectType AT,
                             __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
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

#define declmethod(FUNC, NAME, TYPE)                                           \
    TYPE dpctl_device_##FUNC(__dpctl_keep const DpctlSyclDeviceRef DRef,       \
                             __dpctl_keep const DpctlExecState ES)             \
    {                                                                          \
        auto handler = ES ? dpctl_exec_state_get_error_handler(ES)             \
                          : &DefaultErrorHandler::handler;                     \
        TYPE result = 0;                                                       \
        auto D = unwrap(DRef);                                                 \
        if (D) {                                                               \
            try {                                                              \
                result = D->get_info<info::device::NAME>();                    \
            } catch (runtime_error const &re) {                                \
                std::cerr << re.what() << '\n';                                \
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

bool dpctl_device_get_sub_group_independent_forward_progress(
    __dpctl_keep const DpctlSyclDeviceRef DRef,
    __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
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

uint32_t dpctl_device_get_preferred_vector_width_char(
    __dpctl_keep const DpctlSyclDeviceRef DRef,
    __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
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

uint32_t dpctl_device_get_preferred_vector_width_short(
    __dpctl_keep const DpctlSyclDeviceRef DRef,
    __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
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

uint32_t dpctl_device_get_preferred_vector_width_int(
    __dpctl_keep const DpctlSyclDeviceRef DRef,
    __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
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

uint32_t dpctl_device_get_preferred_vector_width_long(
    __dpctl_keep const DpctlSyclDeviceRef DRef,
    __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
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

uint32_t dpctl_device_get_preferred_vector_width_float(
    __dpctl_keep const DpctlSyclDeviceRef DRef,
    __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
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

uint32_t dpctl_device_get_preferred_vector_width_double(
    __dpctl_keep const DpctlSyclDeviceRef DRef,
    __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
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

uint32_t dpctl_device_get_preferred_vector_width_half(
    __dpctl_keep const DpctlSyclDeviceRef DRef,
    __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
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

__dpctl_give DpctlSyclDeviceRef
dpctl_device_get_parent_device(__dpctl_keep const DpctlSyclDeviceRef DRef,
                               __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
    auto D = unwrap(DRef);
    if (D) {
        try {
            auto parent_D = D->get_info<info::device::parent_device>();
            return wrap(new device(parent_D));
        } catch (invalid_object_error const &ioe) {
            // not a sub device
            return nullptr;
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
            return nullptr;
        }
    }
    else
        return nullptr;
}

__dpctl_give DPCTLDeviceVectorRef dpctl_device_create_sub_devices_equally(
    __dpctl_keep const DpctlSyclDeviceRef DRef,
    size_t count,
    __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
    std::vector<DpctlSyclDeviceRef> *Devices = nullptr;
    if (DRef) {
        if (count == 0) {
            std::cerr << "Can not create sub-devices with zero compute units"
                      << '\n';
            return nullptr;
        }
        auto D = unwrap(DRef);
        try {
            auto subDevices = D->create_sub_devices<
                info::partition_property::partition_equally>(count);
            Devices = new std::vector<DpctlSyclDeviceRef>();
            for (const auto &sd : subDevices) {
                Devices->emplace_back(wrap(new device(sd)));
            }
        } catch (std::bad_alloc const &ba) {
            delete Devices;
            std::cerr << ba.what() << '\n';
            return nullptr;
        } catch (feature_not_supported const &fnse) {
            delete Devices;
            std::cerr << fnse.what() << '\n';
            return nullptr;
        } catch (runtime_error const &re) {
            delete Devices;
            // \todo log error
            std::cerr << re.what() << '\n';
            return nullptr;
        }
    }
    return wrap(Devices);
}

__dpctl_give DPCTLDeviceVectorRef dpctl_device_create_sub_devices_by_counts(
    __dpctl_keep const DpctlSyclDeviceRef DRef,
    __dpctl_keep size_t *counts,
    size_t ncounts,
    __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
    std::vector<DpctlSyclDeviceRef> *Devices = nullptr;
    std::vector<size_t> vcounts(ncounts);
    vcounts.assign(counts, counts + ncounts);
    size_t min_elem = *std::min_element(vcounts.begin(), vcounts.end());
    if (min_elem == 0) {
        std::cerr << "Can not create sub-devices with zero compute units"
                  << '\n';
        return nullptr;
    }
    if (DRef) {
        auto D = unwrap(DRef);
        std::vector<std::remove_pointer<decltype(D)>::type> subDevices;
        try {
            subDevices = D->create_sub_devices<
                info::partition_property::partition_by_counts>(vcounts);
        } catch (feature_not_supported const &fnse) {
            std::cerr << fnse.what() << '\n';
            return nullptr;
        } catch (runtime_error const &re) {
            // \todo log error
            std::cerr << re.what() << '\n';
            return nullptr;
        }
        try {
            Devices = new std::vector<DpctlSyclDeviceRef>();
            for (const auto &sd : subDevices) {
                Devices->emplace_back(wrap(new device(sd)));
            }
        } catch (std::bad_alloc const &ba) {
            delete Devices;
            std::cerr << ba.what() << '\n';
            return nullptr;
        } catch (runtime_error const &re) {
            delete Devices;
            // \todo log error
            std::cerr << re.what() << '\n';
            return nullptr;
        }
    }
    return wrap(Devices);
}

__dpctl_give DPCTLDeviceVectorRef dpctl_device_create_sub_devices_by_affinity(
    __dpctl_keep const DpctlSyclDeviceRef DRef,
    DPCTLPartitionAffinityDomainType PartitionAffinityDomainTy,
    __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
    std::vector<DpctlSyclDeviceRef> *Devices = nullptr;
    auto D = unwrap(DRef);
    if (D) {
        try {
            auto domain = DPCTL_DPCTLPartitionAffinityDomainTypeToSycl(
                PartitionAffinityDomainTy);
            auto subDevices = D->create_sub_devices<
                info::partition_property::partition_by_affinity_domain>(domain);
            Devices = new std::vector<DpctlSyclDeviceRef>();
            for (const auto &sd : subDevices) {
                Devices->emplace_back(wrap(new device(sd)));
            }
        } catch (std::bad_alloc const &ba) {
            delete Devices;
            std::cerr << ba.what() << '\n';
            return nullptr;
        } catch (feature_not_supported const &fnse) {
            delete Devices;
            std::cerr << fnse.what() << '\n';
            return nullptr;
        } catch (runtime_error const &re) {
            delete Devices;
            // \todo log error
            std::cerr << re.what() << '\n';
            return nullptr;
        }
    }
    return wrap(Devices);
}

size_t dpctl_device_hash(__dpctl_keep const DpctlSyclDeviceRef DRef,
                         __dpctl_keep const DpctlExecState ES)
{
    auto handler = ES ? dpctl_exec_state_get_error_handler(ES)
                      : &DefaultErrorHandler::handler;
    if (DRef) {
        auto D = unwrap(DRef);
        std::hash<device> hash_fn;
        return hash_fn(*D);
    }
    else {
        // todo: log error
        std::cerr << "Argument DRef is null"
                  << "/n";
        return 0;
    }
}
