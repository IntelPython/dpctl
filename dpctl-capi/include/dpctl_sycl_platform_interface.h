//===-- dpctl_sycl_platform_interface.h - C API for sycl::platform -*-C++-*- =//
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
/// This header declares a C interface to sycl::platform interface functions.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"
#include "dpctl_data_types.h"
#include "dpctl_sycl_enum_types.h"
#include "dpctl_sycl_platform_manager.h"
#include "dpctl_sycl_types.h"

DPCTL_C_EXTERN_C_BEGIN

/*!
 * @brief Returns a copy of the DPCTLSyclPlatformRef object.
 *
 * @param    DRef           DPCTLSyclPlatformRef object to be copied.
 * @return   A new DPCTLSyclPlatformRef created by copying the passed in
 * DPCTLSyclPlatformRef object.
 */
DPCTL_API
__dpctl_give DPCTLSyclPlatformRef
DPCTLPlatform_Copy(__dpctl_keep const DPCTLSyclPlatformRef PRef);

/*!
 * @brief Creates a new DPCTLSyclPlatformRef for a SYCL platform constructed
 * using SYCL's default_selector.
 *
 * @return   A new DPCTLSyclPlatformRef pointer wrapping a SYCL platform object.
 */
DPCTL_API
__dpctl_give DPCTLSyclPlatformRef DPCTLPlatform_Create();

/*!
 * @brief Creates a new DPCTLSyclPlatformRef for a SYCL platform constructed
 * using the device_selector wrapped by DPCTLSyclDeviceSelectorRef.
 *
 * @param    DSRef          An opaque pointer to a SYCL device_selector object.
 * @return   A new DPCTLSyclPlatformRef pointer wrapping a SYCL platform object.
 */
DPCTL_API
__dpctl_give DPCTLSyclPlatformRef DPCTLPlatform_CreateFromSelector(
    __dpctl_keep const DPCTLSyclDeviceSelectorRef DSRef);

/*!
 * @brief Deletes the DPCTLSyclProgramRef pointer.
 *
 * @param    PRef           An opaque pointer to a sycl::platform.
 */
DPCTL_API
void DPCTLPlatform_Delete(__dpctl_take DPCTLSyclPlatformRef PRef);

/*!
 * @brief  Returns a DPCTLSyclBackendType enum value identifying the SYCL
 * backend associated with the platform.
 *
 * @param    PRef           Opaque pointer to a sycl::platform
 * @return   A DPCTLSyclBackendType enum value identifying the SYCL backend
 * associated with the platform.
 */
DPCTL_API
DPCTLSyclBackendType
DPCTLPlatform_GetBackend(__dpctl_keep const DPCTLSyclPlatformRef PRef);

/*!
 * @brief Returns a C string for the platform name.
 *
 * @param    PRef           Opaque pointer to a sycl::platform
 * @return   A C string containing the name of the sycl::platform.
 */
DPCTL_API
__dpctl_give const char *
DPCTLPlatform_GetName(__dpctl_keep const DPCTLSyclPlatformRef PRef);

/*!
 * @brief Returns a C string corresponding to the vendor providing the platform.
 *
 * @param    PRef           Opaque pointer to a sycl::platform
 * @return    A C string containing the name of the vendor provifing the
 * platform.
 */
DPCTL_API
__dpctl_give const char *
DPCTLPlatform_GetVendor(__dpctl_keep const DPCTLSyclPlatformRef PRef);

/*!
 * @brief Returns the software driver version of the sycl::platform as a C
 * string.
 *
 * @param    PRef           Opaque pointer to a sycl::platform
 * @return   A C string containing the software driver version of the device
 * associated with the platform.
 */
DPCTL_API
__dpctl_give const char *
DPCTLPlatform_GetVersion(__dpctl_keep const DPCTLSyclPlatformRef PRef);

/*!
 * @brief Returns an opaque pointer to a vector of SYCL platforms available on
 * the system.
 *
 * @return    A #DPCTLPlatformVectorRef containing #DPCTLSyclPlatformRef
 * objects.
 */
DPCTL_API
__dpctl_give DPCTLPlatformVectorRef DPCTLPlatform_GetPlatforms();

DPCTL_C_EXTERN_C_END
