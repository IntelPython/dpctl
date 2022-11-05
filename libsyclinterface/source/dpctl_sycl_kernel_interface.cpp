//===--- dpctl_sycl_kernel_interface.cpp - Implements C API for sycl::kernel =//
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
/// This file implements the functions declared in
/// dpctl_sycl_kernel_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_kernel_interface.h"
#include "dpctl_error_handlers.h"
#include "dpctl_string_utils.hpp"
#include "dpctl_sycl_type_casters.hpp"
#include <CL/sycl.hpp> /* Sycl headers */
#include <cstdint>

using namespace sycl;

namespace
{
using namespace dpctl::syclinterface;
} // end of anonymous namespace

size_t DPCTLKernel_GetNumArgs(__dpctl_keep const DPCTLSyclKernelRef KRef)
{
    if (!KRef) {
        error_handler("Cannot get the number of arguments from "
                      "DPCTLSyclKernelRef as input is a nullptr.",
                      __FILE__, __func__, __LINE__);
        return -1;
    }

    auto sycl_kernel = unwrap<kernel>(KRef);
    auto num_args = sycl_kernel->get_info<info::kernel::num_args>();
    return static_cast<size_t>(num_args);
}

void DPCTLKernel_Delete(__dpctl_take DPCTLSyclKernelRef KRef)
{
    delete unwrap<kernel>(KRef);
}

size_t DPCTLKernel_GetWorkGroupSize(__dpctl_keep const DPCTLSyclKernelRef KRef)
{
    if (!KRef) {
        error_handler("Input DPCTKSyclKernelRef is nullptr.", __FILE__,
                      __func__, __LINE__);
        return 0;
    }

    auto sycl_kern = unwrap<kernel>(KRef);
    auto devs = sycl_kern->get_kernel_bundle().get_devices();
    if (devs.empty()) {
        error_handler("Input DPCTKSyclKernelRef has no associated device.",
                      __FILE__, __func__, __LINE__);
        return 0;
    }
    auto v = sycl_kern->get_info<info::kernel_device_specific::work_group_size>(
        devs[0]);
    return static_cast<size_t>(v);
}

size_t DPCTLKernel_GetPreferredWorkGroupSizeMultiple(
    __dpctl_keep const DPCTLSyclKernelRef KRef)
{
    if (!KRef) {
        error_handler("Input DPCTKSyclKernelRef is nullptr.", __FILE__,
                      __func__, __LINE__);
        return 0;
    }

    auto sycl_kern = unwrap<kernel>(KRef);
    auto devs = sycl_kern->get_kernel_bundle().get_devices();
    if (devs.empty()) {
        error_handler("Input DPCTKSyclKernelRef has no associated device.",
                      __FILE__, __func__, __LINE__);
        return 0;
    }
    auto v = sycl_kern->get_info<
        info::kernel_device_specific::preferred_work_group_size_multiple>(
        devs[0]);
    return static_cast<size_t>(v);
}

size_t DPCTLKernel_GetPrivateMemSize(__dpctl_keep const DPCTLSyclKernelRef KRef)
{
    if (!KRef) {
        error_handler("Input DPCTKSyclKernelRef is nullptr.", __FILE__,
                      __func__, __LINE__);
        return 0;
    }

    auto sycl_kern = unwrap<kernel>(KRef);
    auto devs = sycl_kern->get_kernel_bundle().get_devices();
    if (devs.empty()) {
        error_handler("Input DPCTKSyclKernelRef has no associated device.",
                      __FILE__, __func__, __LINE__);
        return 0;
    }
    auto v =
        sycl_kern->get_info<info::kernel_device_specific::private_mem_size>(
            devs[0]);
    return static_cast<size_t>(v);
}

uint32_t
DPCTLKernel_GetMaxNumSubGroups(__dpctl_keep const DPCTLSyclKernelRef KRef)
{
    if (!KRef) {
        error_handler("Input DPCTKSyclKernelRef is nullptr.", __FILE__,
                      __func__, __LINE__);
        return 0;
    }

    auto sycl_kern = unwrap<kernel>(KRef);
    auto devs = sycl_kern->get_kernel_bundle().get_devices();
    if (devs.empty()) {
        error_handler("Input DPCTKSyclKernelRef has no associated device.",
                      __FILE__, __func__, __LINE__);
        return 0;
    }
    auto v =
        sycl_kern->get_info<info::kernel_device_specific::max_num_sub_groups>(
            devs[0]);
    return static_cast<uint32_t>(v);
}

#if 0
// commented out due to bug in DPC++ runtime, get_info for max_sub_group_size
// exported by libsycl has different, not SPEC-compliant signature
uint32_t
DPCTLKernel_GetMaxSubGroupSize(__dpctl_keep const DPCTLSyclKernelRef KRef)
{
    if (!KRef) {
        error_handler("Input DPCTKSyclKernelRef is nullptr.", __FILE__,
                      __func__, __LINE__);
        return 0;
    }

    auto sycl_kern = unwrap<kernel>(KRef);
    auto devs = sycl_kern->get_kernel_bundle().get_devices();
    if (devs.empty()) {
        error_handler("Input DPCTKSyclKernelRef has no associated device.",
                      __FILE__, __func__, __LINE__);
        return 0;
    }
    auto v = sycl_kern
      ->get_info<info::kernel_device_specific::max_sub_group_size>(devs[0]);
    return v;
}
#endif

uint32_t
DPCTLKernel_GetCompileNumSubGroups(__dpctl_keep const DPCTLSyclKernelRef KRef)
{
    if (!KRef) {
        error_handler("Input DPCTKSyclKernelRef is nullptr.", __FILE__,
                      __func__, __LINE__);
        return 0;
    }

    auto sycl_kern = unwrap<kernel>(KRef);
    auto devs = sycl_kern->get_kernel_bundle().get_devices();
    if (devs.empty()) {
        error_handler("Input DPCTKSyclKernelRef has no associated device.",
                      __FILE__, __func__, __LINE__);
        return 0;
    }
    auto v =
        sycl_kern
            ->get_info<info::kernel_device_specific::compile_num_sub_groups>(
                devs[0]);
    return static_cast<uint32_t>(v);
}

uint32_t
DPCTLKernel_GetCompileSubGroupSize(__dpctl_keep const DPCTLSyclKernelRef KRef)
{
    if (!KRef) {
        error_handler("Input DPCTKSyclKernelRef is nullptr.", __FILE__,
                      __func__, __LINE__);
        return 0;
    }

    auto sycl_kern = unwrap<kernel>(KRef);
    auto devs = sycl_kern->get_kernel_bundle().get_devices();
    if (devs.empty()) {
        error_handler("Input DPCTKSyclKernelRef has no associated device.",
                      __FILE__, __func__, __LINE__);
        return 0;
    }
    auto v =
        sycl_kern
            ->get_info<info::kernel_device_specific::compile_sub_group_size>(
                devs[0]);
    return static_cast<uint32_t>(v);
}
