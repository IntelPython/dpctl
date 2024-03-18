//===- dpctl_sycl_kernel_bundle_interface.h - C API for
//     sycl::kernel_bundle<sycl::bundle_state::executable>      -*-C++-*- ===//
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
/// This header declares a C API to create Sycl interoperability programs for
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
 * @defgroup KernelBundleInterface Kernel_bundle class C wrapper
 */

/*!
 * @brief Create a Sycl kernel_bundle from an OpenCL SPIR-V binary file.
 *
 * Uses SYCL2020 interoperability layer to create sycl::kernel_bundle object
 * in executable state for OpenCL and Level-Zero backends from SPIR-V binary.
 *
 * @param    Ctx            An opaque pointer to a sycl::context
 * @param    Dev            An opaque pointer to a sycl::device
 * @param    IL             SPIR-V binary
 * @param    Length         The size of the IL binary in bytes.
 * @param    CompileOpts    Optional compiler flags used when compiling the
 *                          SPIR-V binary.
 * @return   A new SyclKernelBundleRef pointer if the kernel_bundle creation
 * succeeded, else returns NULL.
 * @ingroup KernelBundleInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclKernelBundleRef
DPCTLKernelBundle_CreateFromSpirv(__dpctl_keep const DPCTLSyclContextRef Ctx,
                                  __dpctl_keep const DPCTLSyclDeviceRef Dev,
                                  __dpctl_keep const void *IL,
                                  size_t Length,
                                  const char *CompileOpts);

/*!
 * @brief Create a Sycl kernel bundle from an OpenCL kernel source string.
 *
 * @param    Ctx            An opaque pointer to a sycl::context
 * @param    Dev            An opaque pointer to a sycl::device
 * @param    Source         OpenCL source string
 * @param    CompileOpts    Extra compiler flags (refer Sycl spec.)
 * @return   A new SyclKernelBundleRef pointer if the program creation
 * succeeded, else returns NULL.
 * @ingroup KernelBundleInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclKernelBundleRef DPCTLKernelBundle_CreateFromOCLSource(
    __dpctl_keep const DPCTLSyclContextRef Ctx,
    __dpctl_keep const DPCTLSyclDeviceRef Dev,
    __dpctl_keep const char *Source,
    __dpctl_keep const char *CompileOpts);

/*!
 * @brief Returns the SyclKernel with given name from the program, if not found
 * then return NULL.
 *
 * @param    KBRef          Opaque pointer to a sycl::kernel_bundle
 * @param    KernelName     Name of kernel
 * @return   A SyclKernel reference if the kernel exists, else NULL
 * @ingroup KernelBundleInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclKernelRef
DPCTLKernelBundle_GetKernel(__dpctl_keep DPCTLSyclKernelBundleRef KBRef,
                            __dpctl_keep const char *KernelName);

/*!
 * @brief Return True if a SyclKernel with given name exists in the program, if
 * not found then returns False.
 *
 * @param    KBRef          Opaque pointer to a sycl::kernel_bundle
 * @param    KernelName     Name of kernel
 * @return   True if the kernel exists, else False
 * @ingroup KernelBundleInterface
 */
DPCTL_API
bool DPCTLKernelBundle_HasKernel(__dpctl_keep DPCTLSyclKernelBundleRef KBRef,
                                 __dpctl_keep const char *KernelName);

/*!
 * @brief Frees the DPCTLSyclKernelBundleRef pointer.
 *
 * @param   KBRef           Opaque pointer to a sycl::kernel_bundle
 * @ingroup KernelBundleInterface
 */
DPCTL_API
void DPCTLKernelBundle_Delete(__dpctl_take DPCTLSyclKernelBundleRef KBRef);

/*!
 * @brief Returns a copy of the DPCTLSyclKernelBundleRef object.
 *
 * @param    KBRef           DPCTLSyclKernelBundleRef object to be copied.
 * @return   A new DPCTLSyclKernelBundleRef created by copying the passed in
 * DPCTLSyclKernelBundleRef object.
 * @ingroup KernelBundleInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclKernelBundleRef
DPCTLKernelBundle_Copy(__dpctl_keep const DPCTLSyclKernelBundleRef KBRef);

DPCTL_C_EXTERN_C_END
