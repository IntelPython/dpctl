//===---- dppl_sycl_kernel_interface.h - DPPL-SYCL interface --*--C++ --*--===//
//
//               Python Data Parallel Processing Library (PyDPPL)
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
/// This header declares a C API to create Sycl kernels from OpenCL kernels. In
/// future, API to create interoperability kernels from other languages such as
/// Level-0 driver API may be added here.
///
/// \todo Investigate what we should do when we add support for Level-0 API.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "dppl_data_types.h"
#include "dppl_sycl_types.h"
#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"

DPPL_C_EXTERN_C_BEGIN

/*!
 * @brief Create a Sycl Kernel from an OpenCL SPIR-V binary
 *
 * Sycl 1.2 does expose any method to create a sycl::program from a SPIR-V IL
 * file. To get around this limitation, we need to use the Sycl feature to
 * create an interoperability kernel from an OpenCL kernel. This function first
 * creates an OpenCL program and kernel from the SPIR-V binary and then using
 * the Sycl-OpenCL interoperability feature creates a Sycl kernel from the
 * OpenCL kernel.
 *
 * The feature to create a Sycl kernel from a SPIR-V IL binary will be available
 * in Sycl 2.0.
 *
 * @param    Ctx            An opaque pointer to a sycl::context
 * @param    IL             SPIR-V binary
 * @return   A new SyclProgramRef pointer if the program creation succeeded,
 *           else returns NULL.
 */
DPPL_API
__dppl_give DPPLSyclKernelRef
DPPLKernel_CreateKernelFromSpirv (__dppl_keep const DPPLSyclContextRef Ctx,
                                  __dppl_keep const void *IL,
                                  size_t length,
                                  const char *KernelName = nullptr);

/*!
 * @brief Returns a C string for the kernel name.
 *
 * @param    Kernel         DPPLSyclKernelRef pointer to an OpenCL
 *                          interoperability kernel.
 * @return   If a kernel name exists then returns it as a C string, else
 *           returns a nullptr.
 */
DPPL_API
__dppl_give const char*
DPPLKernel_GetFunctionName (__dppl_keep const DPPLSyclKernelRef Kernel);

/*!
 * @brief Returns the number of arguments for the OpenCL kernel.
 *
 * @param    Kernel         DPPLSyclKernelRef pointer to an OpenCL
 *                          interoperability kernel.
 * @return   Returns the number of arguments for the OpenCL interoperability
 *           kernel.
 */
DPPL_API
size_t
DPPLKernel_GetNumArgs (__dppl_keep const DPPLSyclKernelRef Kernel);

/*!
 * @brief Deletes the DPPLSyclKernelRef after casting it to a sycl::kernel.
 *
 * @param    Kernel         DPPLSyclKernelRef pointer to an OpenCL
 *                          interoperability kernel.
 */
DPPL_API
void
DPPLKernel_DeleteKernelRef (__dppl_take DPPLSyclKernelRef Kernel);

DPPL_C_EXTERN_C_END
