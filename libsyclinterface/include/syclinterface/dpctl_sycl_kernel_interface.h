//===--- dpctl_sycl_kernel_interface.h - C API for sycl::kernel  -*-C++-*- ===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2024 Intel Corporation
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
/// This header declares a C API to create Sycl interoperability kernels for
/// OpenCL and Level Zero driver API.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"
#include "dpctl_data_types.h"
#include "dpctl_sycl_types.h"

DPCTL_C_EXTERN_C_BEGIN

/**
 * @defgroup KernelInterface Kernel class C wrapper
 */

/*!
 * @brief Returns the number of arguments for the sycl
 * interoperability kernel.
 *
 * @param    KRef           DPCTLSyclKernelRef pointer to a SYCL
 *                          interoperability kernel.
 * @return   Returns the number of arguments for the interoperability
 *           kernel.
 * @ingroup KernelInterface
 */
DPCTL_API
size_t DPCTLKernel_GetNumArgs(__dpctl_keep const DPCTLSyclKernelRef KRef);

/*!
 * @brief Deletes the DPCTLSyclKernelRef after casting it to a
 * ``sycl::kernel``.
 *
 * @param    KRef           DPCTLSyclKernelRef pointer to a SYCL
 *                          interoperability kernel.
 * @ingroup KernelInterface
 */
DPCTL_API
void DPCTLKernel_Delete(__dpctl_take DPCTLSyclKernelRef KRef);

/*!
 * @brief Returns a copy of the DPCTLSyclKernelRef object.
 *
 * @param    KRef           DPCTLSyclKernelRef object to be copied.
 * @return   A new DPCTLSyclKernelRef created by copying the passed in
 * DPCTLSyclKernelRef object.
 * @ingroup KernelInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclKernelRef
DPCTLKernel_Copy(__dpctl_keep const DPCTLSyclKernelRef KRef);

/*!
 * !brief Wrapper around
 * `kernel::get_info<info::kernel_device_specific::work_group_size>()`.
 *
 * @param   KRef           DPCTLSyclKernelRef pointer to a SYCL
 *                         interoperability kernel.
 * @return  Returns the maximum number of work-items in a work-group
 *          that can be used to execute a kernel on the device it was
 *          built for.
 * @ingroup KernelInterface
 */
DPCTL_API
size_t DPCTLKernel_GetWorkGroupSize(__dpctl_keep const DPCTLSyclKernelRef KRef);

/*!
 * !brief Wrapper around
 * `kernel::get_info<info::kernel_device_specific::preferred_work_group_size_multiple>()`.
 *
 * @param   KRef           DPCTLSyclKernelRef pointer to a SYCL
 *                         interoperability kernel.
 * @return  Returns a value, of which work-group size is preferred to be a
 * multiple, for executing a kernel on the device it was built for.
 * @ingroup KernelInterface
 */
DPCTL_API
size_t DPCTLKernel_GetPreferredWorkGroupSizeMultiple(
    __dpctl_keep const DPCTLSyclKernelRef KRef);

/*!
 * !brief Wrapper around
 * `kernel::get_info<info::kernel_device_specific::private_mem_size>()`.
 *
 * @param   KRef           DPCTLSyclKernelRef pointer to a SYCL
 *                         interoperability kernel.
 * @return  Returns the minimum amount of private memory, in bytes,
 *          used by each work-item in the kernel.
 * @ingroup KernelInterface
 */
DPCTL_API
size_t
DPCTLKernel_GetPrivateMemSize(__dpctl_keep const DPCTLSyclKernelRef KRef);

/*!
 * !brief Wrapper around
 * `kernel::get_info<info::kernel_device_specific::max_num_sub_groups>()`.
 *
 * @param   KRef           DPCTLSyclKernelRef pointer to an SYCL
 *                         interoperability kernel.
 * @return  Returns the maximum number of sub-groups for this kernel.
 * @ingroup KernelInterface
 */
DPCTL_API
uint32_t
DPCTLKernel_GetMaxNumSubGroups(__dpctl_keep const DPCTLSyclKernelRef KRef);

/*!
 * !brief Wrapper around
 * `kernel::get_info<info::kernel_device_specific::max_sub_group_size>()`.
 *
 * @param   KRef           DPCTLSyclKernelRef pointer to an SYCL
 *                         interoperability kernel.
 * @return  Returns the maximum sub-group size for this kernel.
 * @ingroup KernelInterface
 */
DPCTL_API
uint32_t
DPCTLKernel_GetMaxSubGroupSize(__dpctl_keep const DPCTLSyclKernelRef KRef);

/*!
 * !brief Wrapper around
 * `kernel::get_info<info::kernel_device_specific::compile_num_sub_groups>()`.
 *
 * @param   KRef           DPCTLSyclKernelRef pointer to an SYCL
 *                         interoperability kernel.
 * @return  Returns the number of sub-groups specified by the kernel,
 *          or 0 (if not specified).
 * @ingroup KernelInterface
 */
DPCTL_API
uint32_t
DPCTLKernel_GetCompileNumSubGroups(__dpctl_keep const DPCTLSyclKernelRef KRef);

/*!
 * !brief Wrapper around
 * `kernel::get_info<info::kernel_device_specific::compile_sub_group_size>()`.
 *
 * @param   KRef           DPCTLSyclKernelRef pointer to an SYCL
 *                         interoperability kernel.
 * @return  Returns the required sub-group size specified by this kernel,
 *          or 0 (if not specified).
 * @ingroup KernelInterface
 */
DPCTL_API
uint32_t
DPCTLKernel_GetCompileSubGroupSize(__dpctl_keep const DPCTLSyclKernelRef KRef);

DPCTL_C_EXTERN_C_END
