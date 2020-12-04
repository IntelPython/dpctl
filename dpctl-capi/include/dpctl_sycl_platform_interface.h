//===----------- dpctl_sycl_platform_interface.h - dpctl-C_API ---*--C++ -*-===//
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
/// This header declares a C interface to sycl::platform interface functions.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "dpctl_data_types.h"
#include "dpctl_sycl_enum_types.h"
#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"

DPCTL_C_EXTERN_C_BEGIN

/*!
 * @brief Returns the number of non-host type sycl::platform available on the
 * system.
 *
 * @return The number of available sycl::platforms.
 */
DPCTL_API
size_t DPCTLPlatform_GetNumNonHostPlatforms ();

/*!
 * @brief Returns the number of unique non-host sycl backends on the system.
 *
 * @return   The number of unique sycl backends.
 */
DPCTL_API
size_t DPCTLPlatform_GetNumNonHostBackends ();

/*!
 * @brief Returns an array of the unique non-host DPCTLSyclBackendType values on
 * the system.
 *
 * @return   An array of DPCTLSyclBackendType enum values.
 */
DPCTL_API
__dpctl_give DPCTLSyclBackendType* DPCTLPlatform_GetListOfNonHostBackends ();

/*!
 * @brief Frees an array of DPCTLSyclBackendType enum values.
 *
 * @param    BEArr      An array of DPCTLSyclBackendType enum values to be freed.
 */
DPCTL_API
void DPCTLPlatform_DeleteListOfBackends (__dpctl_take DPCTLSyclBackendType* BEArr);

/*!
 * @brief Prints out some selected info about all sycl::platform on the system.
 *
 */
DPCTL_API
void DPCTLPlatform_DumpInfo ();

DPCTL_C_EXTERN_C_END
