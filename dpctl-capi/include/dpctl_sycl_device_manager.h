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
/// This file declares a set of helper functions to query about the available
/// SYCL devices and backends on the system.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"
#include "dpctl_data_types.h"
#include "dpctl_sycl_types.h"

DPCTL_C_EXTERN_C_BEGIN

/**
 * @defgroup DeviceManager Device management helper functions
 */

/*!
 * @brief Contains a #DPCTLSyclDeviceRef and #DPCTLSyclContextRef 2-tuple that
 * contains a sycl::device and a sycl::context associated with that device.
 */
typedef struct DeviceAndContextPair
{
    DPCTLSyclDeviceRef DRef;
    DPCTLSyclContextRef CRef;
} DPCTL_DeviceAndContextPair;

/*!
 * @brief Checks if two ::DPCTLSyclDeviceRef objects point to the same
 * sycl::device.
 *
 * DPC++ 2021.1.2 has some bugs that prevent the equality of sycl::device
 * objects to work correctly. The DPCTLDeviceMgr_AreEq implements a workaround
 * to check if two sycl::device pointers are equivalent. Since, DPC++ uses
 * std::shared_pointer wrappers for sycl::device objects we check if the raw
 * pointer (shared_pointer.get()) for each device are the same. One caveat is
 * that the trick works only for non-host devices. The function evaluates host
 * devices separately and always assumes that all host devices are equivalent,
 * while checking for the raw pointer equivalent for all other types of devices.
 * The workaround will be removed once DPC++ is fixed to correctly check device
 * equivalence.
 *
 * @param    DRef1          First opaque pointer to a sycl device.
 * @param    DRef2          Second opaque pointer to a sycl device.
 * @return   True if the underlying sycl::device are same, false otherwise.
 * @ingroup DeviceManager
 */
bool DPCTLDeviceMgr_AreEq(__dpctl_keep const DPCTLSyclDeviceRef DRef1,
                          __dpctl_keep const DPCTLSyclDeviceRef DRef2);

/*!
 * @brief Returns an opaque pointer to an empty std::vector<DPCTLSyclDeviceRef>
 * object.
 *
 * The function is useful whenever we want to create a std::vector of
 * sycl::device objects in the C API and then pass the vector to another
 * function. An example is the constructor of sycl::context class that creates a
 * sycl context for a std::vector of sycl::device objects. Using an opaque
 * pointer to an std::vector<DPCTLSyclDeviceRef> object makes it easy to use the
 * container in languages such as Cython that can access C++ types. It allows us
 * to pass through the collection of device objects without having to create a
 * specialized list or array. The DPCTLDeviceVectorRef needs to be only
 * reinterpret_cast into a std::vector and accessed. The
 * DPCTLDeviceMgr_DeleteDeviceVector() or DPCTLDeviceMgr_DeleteDeviceVectorAll()
 * functions should be used to free the ::DPCTLDeviceVectorRef.
 *
 * @return   An opaque pointer to a std::vector<DPCTLSyclDeviceRef> container.
 * The ownership of both the vector and the ::DPCTLSyclDeviceRef elements of the
 * vector are passed to the caller.
 * @ingroup DeviceManager
 */
DPCTL_API
__dpctl_give DPCTLDeviceVectorRef DPCTLDeviceMgr_CreateDeviceVector();

/*!
 * @brief Frees a pointer to a std::vector<DPCTLSyclBackendType>.
 *
 * @param    BVRef          The std::vector<DPCTLSyclBackendType>* that is to
 *                          be freed.
 * @ingroup DeviceManager
 */
DPCTL_API
void DPCTLDeviceMgr_DeleteBackendVector(
    __dpctl_take DPCTLBackendVectorRef BVRef);

/*!
 * @brief Frees a pointer to a std::vector<DPCTLSyclDeviceRef> but does
 * not free the elements of the vector.
 *
 * @param    DVRef          Opaque pointer to a
 *                          std::vector<DPCTLSyclDeviceRef>.
 * @ingroup DeviceManager
 */
DPCTL_API
void DPCTLDeviceMgr_DeleteDeviceVector(__dpctl_take DPCTLDeviceVectorRef DVRef);

/*!
 * @brief Frees a pointer to a std::vector<sycl::DPCTLSyclDeviceRef> and all
 * the elements of the vector.
 *
 * @param    DVRef          Opaque pointer to a
 *                          std::vector<sycl::DPCTLSyclDeviceRef>.
 * @ingroup DeviceManager
 */
DPCTL_API
void DPCTLDeviceMgr_DeviceVector_Clear(__dpctl_take DPCTLDeviceVectorRef DVRef);

/*!
 * @brief Returns an opaque pointer wrapping a std::vector<syc::backend>
 * container that has a list of all the unique backends available on the system.
 *
 * @return   A pointer to a std::vector<syc::backend> storing the list of unique
 * sycl::backend available on the system. Note that a single backend can contain
 * multiple types of devices, *e.g.* the opencl backend.
 * @ingroup DeviceManager
 */
DPCTL_API
__dpctl_give DPCTLBackendVectorRef DPCTLDeviceMgr_GetBackends();

/*!
 * @brief Returns a pointer to a std::vector<sycl::DPCTLSyclDeviceRef>
 * containing the set of ::DPCTLSyclDeviceRef pointers matching the passed in
 * device_identifier bit flag.
 *
 * The device_identifier can be a combination of #DPCTLSyclBackendType and
 * #DPCTLSyclDeviceType bit flags. The function returns all devices that
 * match the specified bit flags. For example,
 *
 *  @code
 *    // Returns all opencl devices
 *    DPCTLDeviceMgr_GetDevices(DPCTLSyclBackendType::DPCTL_OPENCL);
 *
 *    // Returns all opencl gpu devices
 *    DPCTLDeviceMgr_GetDevices(
 *        DPCTLSyclBackendType::DPCTL_OPENCL|DPCTLSyclDeviceType::DPCTL_GPU);
 *
 *    // Returns all gpu devices
 *    DPCTLDeviceMgr_GetDevices(DPCTLSyclDeviceType::DPCTL_GPU);
 *  @endcode
 *
 * @param    device_identifier A bitflag that can be any combination of
 *                             #DPCTLSyclBackendType and #DPCTLSyclDeviceType
 *                             enum values.
 * @return   A #DPCTLDeviceVectorRef containing #DPCTLSyclDeviceRef objects
 * that match the device identifier bit flags.
 * @ingroup DeviceManager
 */
DPCTL_API
__dpctl_give DPCTLDeviceVectorRef
DPCTLDeviceMgr_GetDevices(int device_identifier);

/*!
 * @brief Returns the default sycl context inside an opaque DPCTLSyclContextRef
 * pointer for the DPCTLSyclDeviceRef input argument.
 *
 * @param    DRef           A pointer to a sycl::device that will be used to
 *                          search an internal map containing a cached "default"
 *                          sycl::context for the device.
 * @return   A #DPCTL_DeviceAndContextPair struct containing the cached
 * #DPCTLSyclContextRef associated with the #DPCTLSyclDeviceRef argument passed
 * to the function. The DPCTL_DeviceAndContextPair also contains a
 * #DPCTLSyclDeviceRef pointer pointing to the same device as the input
 * #DPCTLSyclDeviceRef. The returned #DPCTLSyclDeviceRef was cached along with
 * the #DPCTLSyclContextRef. This is a workaround till device equality is
 * properly fixed in DPC++. If the #DPCTLSyclDeviceRef is not found in the cache
 * then DPCTL_DeviceAndContextPair contains a pair of nullptr.
 * @ingroup DeviceManager
 */
DPCTL_API
DPCTL_DeviceAndContextPair DPCTLDeviceMgr_GetDeviceAndContextPair(
    __dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Returns the number of unique sycl::backend available on the system
 *
 * @return   The number of unique sycl::backend on the system. Note
 * that a single backend can have multiple type of devices (e.g. OpenCL).
 * @ingroup DeviceManager
 */
DPCTL_API
size_t DPCTLDeviceMgr_GetNumBackends();

/*!
 * @brief Get the number of available devices for given backend and device type
 * combination.
 *
 * @param    device_identifier Identifies a device using a combination of
 *                             #DPCTLSyclBackendType and #DPCTLSyclDeviceType
 *                             enum values. The argument can be either one of
 *                             the enum values or a bitwise OR-ed combination.
 * @return   The number of available devices satisfying the condition specified
 * by the device_identifier bit flag.
 * @ingroup DeviceManager
 */
DPCTL_API
size_t DPCTLDeviceMgr_GetNumDevices(int device_identifier);

/*!
 * @brief Prints out the info::deivice attributes for the device that are
 * currently supported by dpCtl.
 *
 * @param    DRef           A #DPCTLSyclDeviceRef opaque pointer.
 * @ingroup DeviceManager
 */
DPCTL_API
void DPCTLDeviceMgr_PrintDeviceInfo(__dpctl_keep const DPCTLSyclDeviceRef DRef);

DPCTL_C_EXTERN_C_END
