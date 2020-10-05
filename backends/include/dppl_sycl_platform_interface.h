//===--- dppl_sycl_platform_interface.h - DPPL-SYCL interface ---*--C++ -*-===//
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
/// This header declares a C interface to sycl::platform interface functions.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "dppl_data_types.h"
#include "dppl_sycl_enum_types.h"
#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"

DPPL_C_EXTERN_C_BEGIN

/*!
 * @brief Returns the number of sycl::platform available on the system.
 *
 * @return The number of available sycl::platforms.
 */
DPPL_API
size_t DPPLPlatform_GetNumPlatforms ();

/*!
 * @brief Returns the number of unique sycl backends on the system not counting
 * the host backend.
 *
 * @return   The number of unique sycl backends.
 */
DPPL_API
size_t DPPLPlatform_GetNumBackends ();

/*!
 * @brief Returns an array of the unique DPPLSyclBackendType values on the system.
 *
 * @return   An array of DPPLSyclBackendType enum values.
 */
DPPL_API
__dppl_give DPPLSyclBackendType* DPPLPlatform_GetListOfBackends ();

/*!
 * @brief Frees an array of DPPLSyclBackendType enum values.
 *
 * @param    BEArr      An array of DPPLSyclBackendType enum values to be freed.
 */
DPPL_API
void DPPLPlatform_DeleteListOfBackends (__dppl_take DPPLSyclBackendType* BEArr);

/*!
 * @brief Prints out some selected info about all sycl::platform on the system.
 *
 */
DPPL_API
void DPPLPlatform_DumpInfo ();

DPPL_C_EXTERN_C_END
