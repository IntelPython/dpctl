//===--- dppl_sycl_device_interface.h - DPPL-SYCL interface ---*---C++ -*---===//
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
/// This header declares a C interface to sycl::device. Not all of the device
/// API is exposed, only the bits needed in other places like context and queue
/// interfaces.
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
 * @brief Redefinition of Sycl's device_type so that we do not have to include
 * sycl.hpp here and in the Python bindings.
 *
 */
typedef enum
{
    DPPL_CPU,
    DPPL_GPU,
    DPPL_ACCELERATOR,
    DPPL_CUSTOM,
    DPPL_AUTOMATIC,
    DPPL_HOST,
    DPPL_ALL
} DPPLSyclDeviceType;

/*!
 * @brief Prints out some of the info::deivice attributes for the device.
 *
 * @param    DRef           A DPPLSyclDeviceRef pointer.
 */
DPPL_API
void DPPLDumpDeviceInfo (__dppl_keep const DPPLSyclDeviceRef DRef);

/*!
 * @brief Deletes a DPPLSyclDeviceRef pointer after casting to to sycl::device.
 *
 * @param    DRef           The DPPLSyclDeviceRef pointer to be freed.
 */
DPPL_API
void DPPLDeleteSyclDevice (__dppl_take DPPLSyclDeviceRef DRef);

/*!
 * @brief Returns true if this SYCL device is an OpenCL device and the device
 * type is sycl::info::device_type::accelerator.
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   True if the device type is an accelerator, else False.
 */
DPPL_API
bool DPPLDeviceIsAccelerator (__dppl_keep const DPPLSyclDeviceRef DRef);

/*!
 * @brief Returns true if this SYCL device is an OpenCL device and the device
 * type is sycl::info::device_type::cpu.
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   True if the device type is a cpu, else False.
 */
DPPL_API
bool DPPLDeviceIsCPU (__dppl_keep const DPPLSyclDeviceRef DRef);

/*!
 * @brief Returns true if this SYCL device is an OpenCL device and the device
 * type is sycl::info::device_type::gpu.
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return    True if the device type is a gpu, else False.
 */
DPPL_API
bool DPPLDeviceIsGPU (__dppl_keep const DPPLSyclDeviceRef DRef);

/*!
 * @brief Returns true if this SYCL device is a host device.
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   True if the device is a host device, else False.
 */
DPPL_API
bool DPPLDeviceIsHost (__dppl_keep const DPPLSyclDeviceRef DRef);

/*!
 * @brief Returns the OpenCL software driver version as a C string.
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   A C string in the form major_number.minor.number that corresponds
 *           to the OpenCL driver version if this is a OpenCL device.
 */
DPPL_API
__dppl_give const char*
DPPLGetDeviceDriverInfo (__dppl_keep const DPPLSyclDeviceRef DRef);

/*!
 * @brief Deletes a C string corresponding to the string returned by
 * DPPLGetDeviceDriverInfo().
 *
 * @param    DriverInfo     C String for the driver number.
 */
DPPL_API
void DPPLDeleteDeviceDriverInfo (__dppl_take const char* DriverInfo);

/*!
 * @brief Returns a C string for the device name.
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   A C string containing the OpenCL device name.
 */
DPPL_API
__dppl_give const char*
DPPLGetDeviceName (__dppl_keep const DPPLSyclDeviceRef DRef);

/*!
 * @brief Deletes a C string corresponding to the string returned by
 * DPPLGetDeviceName().
 *
 * @param    DeviceName     C String for the device name.
 */
DPPL_API
void DPPLDeleteDeviceName (__dppl_take const char* DeviceName);

/*!
 * @brief Returns a C string corresponding to the vendor name.
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return    A C string containing the OpenCL device vendor name.
 */
DPPL_API
__dppl_give const char*
DPPLGetDeviceVendorName (__dppl_keep const DPPLSyclDeviceRef DRef);

/*!
 * @brief Deletes a C string corresponding to the string returned by
 * DPPLGetDeviceVendorName().
 *
 * @param    char           C String for the vendor name.
 */
DPPL_API
void DPPLDeleteDeviceVendorName (__dppl_take const char* VendorName);

/*!
 * @brief Returns True if the device and the host share a unified memory
 * subsystem, else returns False.
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   Boolean indicating if the device shares a unified memory subsystem
 * with the host.
 */
DPPL_API
bool DPPLGetDeviceHostUnifiedMemory (__dppl_keep const DPPLSyclDeviceRef DRef);

DPPL_C_EXTERN_C_END
