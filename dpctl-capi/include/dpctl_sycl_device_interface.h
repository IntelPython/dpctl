//===----- dpctl_sycl_device_interface.h - C API for sycl::device -*-C++-*- ==//
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
/// This header declares a C interface to sycl::device. Not all of the device
/// API is exposed, only the bits needed in other places like context and queue
/// interfaces.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"
#include "dpctl_data_types.h"
#include "dpctl_sycl_enum_types.h"
#include "dpctl_sycl_types.h"

DPCTL_C_EXTERN_C_BEGIN

/*!
 * @brief Returns a copy of the DPCTLSyclDeviceRef object.
 *
 * @param    DRef           DPCTLSyclDeviceRef object to be copied.
 * @return   A new DPCTLSyclDeviceRef created by copying the passed in
 * DPCTLSyclDeviceRef object.
 */
DPCTL_API
__dpctl_give DPCTLSyclDeviceRef
DPCTLDevice_Copy(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Returns a new DPCTLSyclDeviceRef opaque object wrapping a SYCL device
 * instance as a host device.
 *
 * @return   An opaque pointer to the host SYCL device.
 */
DPCTL_API
__dpctl_give DPCTLSyclDeviceRef DPCTLDevice_Create();

/*!
 * @brief Returns a new DPCTLSyclDeviceRef opaque object created using the
 * provided device_selector.
 *
 * @param    DSRef          An opaque pointer to a SYCL device_selector.
 * @return   Returns an opaque pointer to a SYCL device created using the
 *           device_selector, if the requested device could not be created a
 *           nullptr is returned.
 */
DPCTL_API
__dpctl_give DPCTLSyclDeviceRef DPCTLDevice_CreateFromSelector(
    __dpctl_keep const DPCTLSyclDeviceSelectorRef DSRef);

/*!
 * @brief Prints out some of the info::deivice attributes for the device.
 *
 * @param    DRef           A DPCTLSyclDeviceRef pointer.
 */
DPCTL_API
void DPCTLDevice_DumpInfo(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Deletes a DPCTLSyclDeviceRef pointer after casting to to sycl::device.
 *
 * @param    DRef           The DPCTLSyclDeviceRef pointer to be freed.
 */
DPCTL_API
void DPCTLDevice_Delete(__dpctl_take DPCTLSyclDeviceRef DRef);

/*!
 * @brief Returns true if this SYCL device is an OpenCL device and the device
 * type is sycl::info::device_type::accelerator.
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   True if the device type is an accelerator, else False.
 */
DPCTL_API
bool DPCTLDevice_IsAccelerator(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Returns true if this SYCL device is an OpenCL device and the device
 * type is sycl::info::device_type::cpu.
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   True if the device type is a cpu, else False.
 */
DPCTL_API
bool DPCTLDevice_IsCPU(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Returns true if this SYCL device is an OpenCL device and the device
 * type is sycl::info::device_type::gpu.
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return    True if the device type is a gpu, else False.
 */
DPCTL_API
bool DPCTLDevice_IsGPU(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Returns true if this SYCL device is a host device.
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   True if the device is a host device, else False.
 */
DPCTL_API
bool DPCTLDevice_IsHost(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Returns the OpenCL software driver version as a C string.
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   A C string in the form major_number.minor.number that corresponds
 *           to the OpenCL driver version if this is a OpenCL device.
 */
DPCTL_API
__dpctl_give const char *
DPCTLDevice_GetDriverInfo(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over device.get_info<info::device::max_compute_units>().
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   Returns the valid result if device exists else returns 0.
 */
DPCTL_API
uint32_t
DPCTLDevice_GetMaxComputeUnits(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper for get_info<info::device::max_work_item_dimensions>().
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   Returns the valid result if device exists else returns 0.
 */
DPCTL_API
uint32_t
DPCTLDevice_GetMaxWorkItemDims(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper for get_info<info::device::max_work_item_sizes>().
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   Returns the valid result if device exists else returns NULL.
 */
DPCTL_API
__dpctl_keep size_t *
DPCTLDevice_GetMaxWorkItemSizes(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper for get_info<info::device::max_work_group_size>().
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   Returns the valid result if device exists else returns 0.
 */
DPCTL_API
size_t
DPCTLDevice_GetMaxWorkGroupSize(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over device.get_info<info::device::max_num_sub_groups>.
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   Returns the valid result if device exists else returns 0.
 */
DPCTL_API
uint32_t
DPCTLDevice_GetMaxNumSubGroups(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * device.get_info<info::device::aspect::int64_base_atomics>.
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   Returns true if device has int64_base_atomics else returns false.
 */
DPCTL_API
bool DPCTLDevice_HasInt64BaseAtomics(
    __dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * device.get_info<info::device::aspect::int64_extended_atomics>.
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   Returns true if device has int64_extended_atomics else returns
 * false.
 */
DPCTL_API
bool DPCTLDevice_HasInt64ExtendedAtomics(
    __dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Returns a C string for the device name.
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   A C string containing the OpenCL device name.
 */
DPCTL_API
__dpctl_give const char *
DPCTLDevice_GetName(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Returns a C string corresponding to the vendor name.
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return    A C string containing the OpenCL device vendor name.
 */
DPCTL_API
__dpctl_give const char *
DPCTLDevice_GetVendorName(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Returns True if the device and the host share a unified memory
 * subsystem, else returns False.
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   Boolean indicating if the device shares a unified memory subsystem
 * with the host.
 */
DPCTL_API
bool DPCTLDevice_IsHostUnifiedMemory(
    __dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Checks if two DPCTLSyclDeviceRef objects point to the same
 * sycl::device.
 *
 * @param    DevRef1       First opaque pointer to the sycl device.
 * @param    DevRef2       Second opaque pointer to the sycl device.
 * @return   True if the underlying sycl::device are same, false otherwise.
 */
DPCTL_API
bool DPCTLDevice_AreEq(__dpctl_keep const DPCTLSyclDeviceRef DevRef1,
                       __dpctl_keep const DPCTLSyclDeviceRef DevRef2);

/*!
 * @brief Checks if device has aspect.
 *
 * @param    DRef       Opaque pointer to a sycl::device
 * @param    AT         DPCTLSyclAspectType of device::aspect.
 * @return   True if sycl::device has device::aspect, else false.
 */
DPCTL_API
bool DPCTLDevice_HasAspect(__dpctl_keep const DPCTLSyclDeviceRef DRef,
                           __dpctl_keep const DPCTLSyclAspectType AT);

DPCTL_C_EXTERN_C_END
