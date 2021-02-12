//===--- dpctl_sycl_module_interface.h - C API for sycl::module  -*-C++-*- ===//
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
/// C API for sycl::module (currently sycl::program) functions.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"
#include "dpctl_data_types.h"
#include "dpctl_sycl_types.h"

DPCTL_C_EXTERN_C_BEGIN

/*!
 * @brief Returns the SyclKernel with given name from the program, if not found
 * then return NULL.
 *
 * @param    PRef           Opaque pointer to a sycl::program
 * @param    KernelName     Name of kernel
 * @return   A SyclKernel reference if the kernel exists, else NULL
 */
DPCTL_API
__dpctl_give DPCTLSyclKernelRef
DPCTLProgram_GetKernel(__dpctl_keep DPCTLSyclProgramRef PRef,
                       __dpctl_keep const char *KernelName);

/*!
 * @brief Return True if a SyclKernel with given name exists in the program, if
 * not found then returns False.
 *
 * @param    PRef           Opaque pointer to a sycl::program
 * @param    KernelName     Name of kernel
 * @return   True if the kernel exists, else False
 */
DPCTL_API
bool DPCTLProgram_HasKernel(__dpctl_keep DPCTLSyclProgramRef PRef,
                            __dpctl_keep const char *KernelName);

/*!
 * @brief Frees the DPCTLSyclProgramRef pointer.
 *
 * @param    PRef           Opaque pointer to a sycl::program
 */
DPCTL_API
void DPCTLProgram_Delete(__dpctl_take DPCTLSyclProgramRef PRef);

DPCTL_C_EXTERN_C_END
