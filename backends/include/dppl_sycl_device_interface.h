//===---------- dppl_sycl_device_interface.h - dpctl-C_API ---*---C++ -*---===//
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
/// This header declares a C interface to sycl::device. Not all of the device
/// API is exposed, only the bits needed in other places like context and queue
/// interfaces.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "dppl_data_types.h"
#include "dppl_sycl_enum_types.h"
#include "dppl_sycl_types.h"
#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"

DPPL_C_EXTERN_C_BEGIN

/*!
 * @brief Prints out some of the info::deivice attributes for the device.
 *
 * @param    DRef           A DPPLSyclDeviceRef pointer.
 */
DPPL_API
void DPPLDevice_DumpInfo (__dppl_keep const DPPLSyclDeviceRef DRef);

/*!
 * @brief Deletes a DPPLSyclDeviceRef pointer after casting to to sycl::device.
 *
 * @param    DRef           The DPPLSyclDeviceRef pointer to be freed.
 */
DPPL_API
void DPPLDevice_Delete (__dppl_take DPPLSyclDeviceRef DRef);

/*!
 * @brief Returns true if this SYCL device is an OpenCL device and the device
 * type is sycl::info::device_type::accelerator.
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   True if the device type is an accelerator, else False.
 */
DPPL_API
bool DPPLDevice_IsAccelerator (__dppl_keep const DPPLSyclDeviceRef DRef);

/*!
 * @brief Returns true if this SYCL device is an OpenCL device and the device
 * type is sycl::info::device_type::cpu.
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   True if the device type is a cpu, else False.
 */
DPPL_API
bool DPPLDevice_IsCPU (__dppl_keep const DPPLSyclDeviceRef DRef);

/*!
 * @brief Returns true if this SYCL device is an OpenCL device and the device
 * type is sycl::info::device_type::gpu.
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return    True if the device type is a gpu, else False.
 */
DPPL_API
bool DPPLDevice_IsGPU (__dppl_keep const DPPLSyclDeviceRef DRef);

/*!
 * @brief Returns true if this SYCL device is a host device.
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   True if the device is a host device, else False.
 */
DPPL_API
bool DPPLDevice_IsHost (__dppl_keep const DPPLSyclDeviceRef DRef);

/*!
 * @brief Returns the OpenCL software driver version as a C string.
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   A C string in the form major_number.minor.number that corresponds
 *           to the OpenCL driver version if this is a OpenCL device.
 */
DPPL_API
__dppl_give const char*
DPPLDevice_GetDriverInfo (__dppl_keep const DPPLSyclDeviceRef DRef);

/*!
 * @brief Returns a C string for the device name.
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   A C string containing the OpenCL device name.
 */
DPPL_API
__dppl_give const char*
DPPLDevice_GetName (__dppl_keep const DPPLSyclDeviceRef DRef);

/*!
 * @brief Returns a C string corresponding to the vendor name.
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return    A C string containing the OpenCL device vendor name.
 */
DPPL_API
__dppl_give const char*
DPPLDevice_GetVendorName (__dppl_keep const DPPLSyclDeviceRef DRef);

/*!
 * @brief Returns True if the device and the host share a unified memory
 * subsystem, else returns False.
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   Boolean indicating if the device shares a unified memory subsystem
 * with the host.
 */
DPPL_API
bool DPPLDevice_IsHostUnifiedMemory (__dppl_keep const DPPLSyclDeviceRef DRef);

DPPL_C_EXTERN_C_END
