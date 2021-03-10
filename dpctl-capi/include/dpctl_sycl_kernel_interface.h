//===--- dpctl_sycl_kernel_interface.h - C API for sycl::kernel  -*-C++-*- ===//
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

/*!
 * @brief Returns a C string for the kernel name.
 *
 * @param    KRef           DPCTLSyclKernelRef pointer to an OpenCL
 *                          interoperability kernel.
 * @return   If a kernel name exists then returns it as a C string, else
 *           returns a nullptr.
 */
DPCTL_API
__dpctl_give const char *
DPCTLKernel_GetFunctionName(__dpctl_keep const DPCTLSyclKernelRef KRef);

/*!
 * @brief Returns the number of arguments for the OpenCL kernel.
 *
 * @param    KRef           DPCTLSyclKernelRef pointer to an OpenCL
 *                          interoperability kernel.
 * @return   Returns the number of arguments for the OpenCL interoperability
 *           kernel.
 */
DPCTL_API
size_t DPCTLKernel_GetNumArgs(__dpctl_keep const DPCTLSyclKernelRef KRef);

/*!
 * @brief Deletes the DPCTLSyclKernelRef after casting it to a sycl::kernel.
 *
 * @param    KRef           DPCTLSyclKernelRef pointer to an OpenCL
 *                          interoperability kernel.
 */
DPCTL_API
void DPCTLKernel_Delete(__dpctl_take DPCTLSyclKernelRef KRef);

DPCTL_C_EXTERN_C_END
