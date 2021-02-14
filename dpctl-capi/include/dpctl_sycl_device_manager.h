//===-- dpctl_sycl_device_manager.h - A manager for sycl devices -*-C++-*- ===//
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
/// Declarations for helper functions to access sycl devices.
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
 * @brief
 *
 */
typedef struct DeviceAndContextPair
{
    DPCTLSyclDeviceRef DRef;
    DPCTLSyclContextRef CRef;
} DPCTL_DeviceAndContextPair;

/*!
 * @brief Opaque pointer to a vector of DPCTLSyclBackendType enums.
 */
typedef struct DPCTLBackendVector *DPCTLBackendVectorRef;

/*!
 * @brief Opaque pointer to a vector of DPCTLSyclDeviceRefs.
 */
typedef struct DPCTLDeviceVector *DPCTLDeviceVectorRef;

/*!
 * @brief
 *
 * @param    BVRef          My Param doc
 * @return   {return}       My Param doc
 */
DPCTL_API
void DPCTLDeviceMgr_DeleteBackendVector(
    __dpctl_take DPCTLBackendVectorRef BVRef);

/*!
 * @brief
 *
 * @param    DVRef          My Param doc
 * @return   {return}       My Param doc
 */
DPCTL_API
void DPCTLDeviceMgr_DeleteDeviceVector(__dpctl_take DPCTLDeviceVectorRef DVRef);

/*!
 * @brief
 *
 * @param    DVRef          My Param doc
 * @return   {return}       My Param doc
 */
DPCTL_API
void DPCTLDeviceMgr_DeleteDeviceVectorAll(
    __dpctl_take DPCTLDeviceVectorRef DVRef);

/*!
 * @brief
 *
 * @return   {return}       My Param doc
 */
DPCTL_API
__dpctl_give DPCTLBackendVectorRef DPCTLDeviceMgr_GetBackends();

/*!
 * @brief
 *
 * @param    device_identifier My Param doc
 * @return   {return}       My Param doc
 */
DPCTL_API
__dpctl_give DPCTLDeviceVectorRef
DPCTLDeviceMgr_GetDevices(int device_identifier);

/*!
 * @brief
 *
 * @param    DRef           My Param doc
 * @return   {return}       My Param doc
 */
DPCTL_API
DPCTL_DeviceAndContextPair DPCTLDeviceMgr_GetDeviceAndContextPair(
    __dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief
 *
 * @return   {return}       My Param doc
 */
DPCTL_API
size_t DPCTLDeviceMgr_GetNumBackends();

/*!
 * @brief Get the number of available devices for given backend and device type
 * combination.
 *
 * @param    device_identifier Identifies a device using a combination of
 *                             DPCTLSyclBackendType and DPCTLSyclDeviceType
 *                             enum values. The argument can be either one of
 *                             the enum values or a bitwise OR-ed combination.
 * @return   The number of available queues.
 */
DPCTL_API
size_t DPCTLDeviceMgr_GetNumDevices(int device_identifier);

/*!
 * @brief Prints out some of the info::deivice attributes for the device.
 *
 * @param    DRef           A DPCTLSyclDeviceRef pointer.
 */
DPCTL_API
void DPCTLDeviceMgr_PrintDeviceInfo(__dpctl_keep const DPCTLSyclDeviceRef DRef);

DPCTL_C_EXTERN_C_END
