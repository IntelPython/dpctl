//===-- dpctl_sycl_platform_interface.h - C API for sycl::platform -*-C++-*- =//
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

/**
 * @defgroup PlatformInterface Platform class C wrapper
 */

/*!
 * @brief Checks if two DPCTLSyclPlatformRef objects point to the same
 * sycl::platform.
 *
 * @param    PRef1         First opaque pointer to a ``sycl::platform``.
 * @param    PRef2         Second opaque pointer to a ``sycl::platform``.
 * @return   True if the underlying sycl::platform are same, false otherwise.
 * @ingroup PlatformInterface
 */
DPCTL_API
bool DPCTLPlatform_AreEq(__dpctl_keep const DPCTLSyclPlatformRef PRef1,
                         __dpctl_keep const DPCTLSyclPlatformRef PRef2);

/*!
 * @brief Returns a copy of the DPCTLSyclPlatformRef object.
 *
 * @param    PRef           DPCTLSyclPlatformRef object to be copied.
 * @return   A new DPCTLSyclPlatformRef created by copying the passed in
 * DPCTLSyclPlatformRef object.
 * @ingroup PlatformInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclPlatformRef
DPCTLPlatform_Copy(__dpctl_keep const DPCTLSyclPlatformRef PRef);

/*!
 * @brief Creates a new DPCTLSyclPlatformRef for a SYCL platform constructed
 * using SYCL's default_selector.
 *
 * @return   A new DPCTLSyclPlatformRef pointer wrapping a SYCL platform object.
 * @ingroup PlatformInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclPlatformRef DPCTLPlatform_Create(void);

/*!
 * @brief Creates a new DPCTLSyclPlatformRef for a SYCL platform constructed
 * using the dpctl_device_selector wrapped by DPCTLSyclDeviceSelectorRef.
 *
 * @param    DSRef          An opaque pointer to a SYCL dpctl_device_selector
 * object.
 * @return   A new DPCTLSyclPlatformRef pointer wrapping a SYCL platform object.
 * @ingroup PlatformInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclPlatformRef DPCTLPlatform_CreateFromSelector(
    __dpctl_keep const DPCTLSyclDeviceSelectorRef DSRef);

/*!
 * @brief Deletes the DPCTLSyclProgramRef pointer.
 *
 * @param    PRef           An opaque pointer to a sycl::platform.
 * @ingroup PlatformInterface
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
 * @ingroup PlatformInterface
 */
DPCTL_API
DPCTLSyclBackendType
DPCTLPlatform_GetBackend(__dpctl_keep const DPCTLSyclPlatformRef PRef);

/*!
 * @brief Returns a C string for the platform name.
 *
 * @param    PRef           Opaque pointer to a sycl::platform
 * @return   A C string containing the name of the sycl::platform.
 * @ingroup PlatformInterface
 */
DPCTL_API
__dpctl_give const char *
DPCTLPlatform_GetName(__dpctl_keep const DPCTLSyclPlatformRef PRef);

/*!
 * @brief Returns a C string corresponding to the vendor providing the platform.
 *
 * @param    PRef           Opaque pointer to a sycl::platform
 * @return    A C string containing the name of the vendor providing the
 * platform.
 * @ingroup PlatformInterface
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
 * @ingroup PlatformInterface
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
 * @ingroup PlatformInterface
 */
DPCTL_API
__dpctl_give DPCTLPlatformVectorRef DPCTLPlatform_GetPlatforms(void);

/*!
 * @brief  Returns a DPCTLSyclContextRef for default platform context.
 *
 * @param    PRef           Opaque pointer to a sycl::platform
 * @return   A DPCTLSyclContextRef value for the default platform associated
 * with this platform.
 * @ingroup PlatformInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclContextRef
DPCTLPlatform_GetDefaultContext(__dpctl_keep const DPCTLSyclPlatformRef PRef);

/*!
 * @brief Wrapper over std::hash<sycl::platform>'s operator()
 *
 * @param    PRef        The DPCTLSyclPlatformRef pointer.
 * @return   Hash value of the underlying ``sycl::platform`` instance.
 * @ingroup PlatformInterface
 */
DPCTL_API
size_t DPCTLPlatform_Hash(__dpctl_keep DPCTLSyclPlatformRef PRef);

DPCTL_C_EXTERN_C_END
