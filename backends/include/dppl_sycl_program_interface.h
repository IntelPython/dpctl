//===---- dppl_sycl_program_interface.h - DPPL-SYCL interface --*--C++ --*--===//
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
/// This header declares a C API to create Sycl program an interoperability
/// program defined in OpenCL. In future, API to create interoperability
/// kernels from other languages such as Level-0 driver API may be added here.
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
 * @brief Create a Sycl program from an OpenCL SPIR-V binary file.
 *
 * Sycl 1.2 does expose any method to create a sycl::program from a SPIR-V IL
 * file. To get around this limitation, we need to use the Sycl feature to
 * create an interoperability kernel from an OpenCL kernel. This function first
 * creates an OpenCL program and kernel from the SPIR-V binary and then using
 * the Sycl-OpenCL interoperability feature creates a Sycl kernel from the
 * OpenCL kernel.
 *
 * The feature to create a Sycl kernel from a SPIR-V IL binary will be available
 * in Sycl 2.0, at which point we may deprecate this function.
 *
 * @param    Ctx            An opaque pointer to a sycl::context
 * @param    IL             SPIR-V binary
 * @return   A new SyclProgramRef pointer if the program creation succeeded,
 *           else returns NULL.
 */
DPPL_API
__dppl_give DPPLSyclProgramRef
DPPLProgram_CreateFromOCLSpirv (__dppl_keep const DPPLSyclContextRef Ctx,
                                __dppl_keep const void *IL,
                                size_t Length);

/*!
 * @brief Create a Sycl program from an OpenCL kernel source string.
 *
 * @param    Ctx            An opaque pointer to a sycl::context
 * @param    Source         OpenCL source string
 * @param    CompileOptions Extra compiler flags (refer Sycl spec.)
 * @return   A new SyclProgramRef pointer if the program creation succeeded,
 *           else returns NULL.
 */
DPPL_API
__dppl_give DPPLSyclProgramRef
DPPLProgram_CreateFromOCLSource (__dppl_keep const DPPLSyclContextRef Ctx,
                                 __dppl_keep const char *Source,
                                 __dppl_keep const char *CompileOpts = nullptr);

/*!
 * @brief Returns the SyclKernel with given name from the program, if not found
 * then return NULL.
 *
 * @param    PRef           Opaque pointer to a sycl::program
 * @param    KernelName     Name of kernel
 * @return   A SyclKernel reference if the kernel exists, else NULL
 */
DPPL_API
__dppl_give DPPLSyclKernelRef
DPPLProgram_GetKernel (__dppl_keep DPPLSyclProgramRef PRef,
                       __dppl_keep const char *KernelName);

/*!
 * @brief Return True if a SyclKernel with given name exists in the program, if
 * not found then returns False.
 *
 * @param    PRef           Opaque pointer to a sycl::program
 * @param    KernelName     Name of kernel
 * @return   True if the kernel exists, else False
 */
DPPL_API
bool
DPPLProgram_HasKernel (__dppl_keep DPPLSyclProgramRef PRef,
                       __dppl_keep const char *KernelName);

/*!
 * @brief Frees the DPPLSyclProgramRef pointer.
 *
 * @param    PRef           Opaque pointer to a sycl::program
 */
DPPL_API
void
DPPLProgram_Delete (__dppl_take DPPLSyclProgramRef PRef);

DPPL_C_EXTERN_C_END
