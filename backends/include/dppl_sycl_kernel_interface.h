//===---- dppl_sycl_kernel_interface.h - DPPL-SYCL interface --*--C++ --*--===//
//
//               Data Parallel Control Library (dpCtl)
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
 * @brief Returns a C string for the kernel name.
 *
 * @param    KRef           DPPLSyclKernelRef pointer to an OpenCL
 *                          interoperability kernel.
 * @return   If a kernel name exists then returns it as a C string, else
 *           returns a nullptr.
 */
DPPL_API
__dppl_give const char*
DPPLKernel_GetFunctionName (__dppl_keep const DPPLSyclKernelRef KRef);

/*!
 * @brief Returns the number of arguments for the OpenCL kernel.
 *
 * @param    KRef           DPPLSyclKernelRef pointer to an OpenCL
 *                          interoperability kernel.
 * @return   Returns the number of arguments for the OpenCL interoperability
 *           kernel.
 */
DPPL_API
size_t
DPPLKernel_GetNumArgs (__dppl_keep const DPPLSyclKernelRef KRef);

/*!
 * @brief Deletes the DPPLSyclKernelRef after casting it to a sycl::kernel.
 *
 * @param    KRef           DPPLSyclKernelRef pointer to an OpenCL
 *                          interoperability kernel.
 */
DPPL_API
void
DPPLKernel_Delete (__dppl_take DPPLSyclKernelRef KRef);

DPPL_C_EXTERN_C_END
