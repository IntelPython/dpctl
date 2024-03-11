//===----- dpctl_sycl_device_interface.h - C API for sycl::device -*-C++-*- ==//
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
#include "dpctl_sycl_device_manager.h"
#include "dpctl_sycl_enum_types.h"
#include "dpctl_sycl_types.h"

DPCTL_C_EXTERN_C_BEGIN

/**
 * @defgroup DeviceInterface Device class C wrapper
 */

/*!
 * @brief Returns a copy of the DPCTLSyclDeviceRef object.
 *
 * @param    DRef           DPCTLSyclDeviceRef object to be copied.
 * @return   A new DPCTLSyclDeviceRef created by copying the passed in
 * DPCTLSyclDeviceRef object.
 * @ingroup DeviceInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclDeviceRef
DPCTLDevice_Copy(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Returns a new DPCTLSyclDeviceRef opaque object wrapping a SYCL device
 * instance as a host device.
 *
 * @return   An opaque pointer to a ``sycl::device`` created as an instance of
 * the host device.
 * @ingroup DeviceInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclDeviceRef DPCTLDevice_Create(void);

/*!
 * @brief Returns a new DPCTLSyclDeviceRef opaque object created using the
 * provided device_selector.
 *
 * @param    DSRef          An opaque pointer to a ``sycl::device_selector``.
 * @return   Returns an opaque pointer to a SYCL device created using the
 *           device_selector, if the requested device could not be created a
 *           nullptr is returned.
 * @ingroup DeviceInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclDeviceRef DPCTLDevice_CreateFromSelector(
    __dpctl_keep const DPCTLSyclDeviceSelectorRef DSRef);

/*!
 * @brief Deletes a DPCTLSyclDeviceRef pointer after casting to to sycl::device.
 *
 * @param    DRef           The DPCTLSyclDeviceRef pointer to be freed.
 * @ingroup DeviceInterface
 */
DPCTL_API
void DPCTLDevice_Delete(__dpctl_take DPCTLSyclDeviceRef DRef);

/*!
 * @brief Returns true if this SYCL device is an OpenCL device and the device
 * type is ``sycl::info::device_type::accelerator``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   True if the device type is an accelerator, else False.
 * @ingroup DeviceInterface
 */
DPCTL_API
bool DPCTLDevice_IsAccelerator(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Returns true if this SYCL device is an OpenCL device and the device
 * type is ``sycl::info::device_type::cpu``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   True if the device type is a cpu, else False.
 * @ingroup DeviceInterface
 */
DPCTL_API
bool DPCTLDevice_IsCPU(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Returns true if this SYCL device is an OpenCL device and the device
 * type is ``sycl::info::device_type::gpu``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return    True if the device type is a gpu, else False.
 * @ingroup DeviceInterface
 */
DPCTL_API
bool DPCTLDevice_IsGPU(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Returns the backend for the device.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   A DPCTLSyclBackendType enum value representing the
 * ``sycl::backend`` for the device.
 * @ingroup DeviceInterface
 */
DPCTL_API
DPCTLSyclBackendType
DPCTLDevice_GetBackend(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Returns the DPCTLSyclDeviceType enum value for the DPCTLSyclDeviceRef
 * argument.
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   The DPCTLSyclDeviceType value corresponding to the device.
 * @ingroup DeviceInterface
 */
DPCTL_API
DPCTLSyclDeviceType
DPCTLDevice_GetDeviceType(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Returns the OpenCL software driver version as a C string.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   A C string in the form major_number.minor.number that corresponds
 *           to the OpenCL driver version if this is a OpenCL device.
 * @ingroup DeviceInterface
 */
DPCTL_API
__dpctl_give const char *
DPCTLDevice_GetDriverVersion(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over device.get_info<info::device::max_compute_units>().
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   Returns the valid result if device exists else returns 0.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint32_t
DPCTLDevice_GetMaxComputeUnits(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over device.get_info<info::device::global_mem_size>().
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   Returns the valid result if device exists else returns 0.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint64_t
DPCTLDevice_GetGlobalMemSize(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over device.get_info<info::device::local_mem_size>().
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   Returns the valid result if device exists else returns 0.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint64_t
DPCTLDevice_GetLocalMemSize(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper for get_info<info::device::max_work_item_dimensions>().
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   Returns the valid result if device exists else returns 0.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint32_t
DPCTLDevice_GetMaxWorkItemDims(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper for get_info<info::device::max_work_item_sizes<1>>().
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   Returns the valid result if device exists else returns NULL.
 * @ingroup DeviceInterface
 */
DPCTL_API
__dpctl_keep size_t *
DPCTLDevice_GetMaxWorkItemSizes1d(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper for get_info<info::device::max_work_item_sizes<2>>().
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   Returns the valid result if device exists else returns NULL.
 * @ingroup DeviceInterface
 */
DPCTL_API
__dpctl_keep size_t *
DPCTLDevice_GetMaxWorkItemSizes2d(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper for get_info<info::device::max_work_item_sizes<3>>().
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   Returns the valid result if device exists else returns NULL.
 * @ingroup DeviceInterface
 */
DPCTL_API
__dpctl_keep size_t *
DPCTLDevice_GetMaxWorkItemSizes3d(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper for get_info<info::device::max_work_group_size>().
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   Returns the valid result if device exists else returns 0.
 * @ingroup DeviceInterface
 */
DPCTL_API
size_t
DPCTLDevice_GetMaxWorkGroupSize(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over device.get_info<info::device::max_num_sub_groups>.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   Returns the valid result if device exists else returns 0.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint32_t
DPCTLDevice_GetMaxNumSubGroups(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Returns the ``sycl::platform`` for the device as DPCTLSyclPlatformRef
 * opaque pointer.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   An opaque pointer to the sycl::platform for the device.
 * @ingroup DeviceInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclPlatformRef
DPCTLDevice_GetPlatform(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Returns a C string for the device name.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   A C string containing the OpenCL device name.
 * @ingroup DeviceInterface
 */
DPCTL_API
__dpctl_give const char *
DPCTLDevice_GetName(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Returns a C string corresponding to the vendor name.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return    A C string containing the OpenCL device vendor name.
 * @ingroup DeviceInterface
 */
DPCTL_API
__dpctl_give const char *
DPCTLDevice_GetVendor(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Checks if two DPCTLSyclDeviceRef objects point to the same
 * sycl::device.
 *
 * @param    DRef1         First opaque pointer to a ``sycl::device``.
 * @param    DRef2         Second opaque pointer to a ``sycl::device``.
 * @return   True if the underlying sycl::device are same, false otherwise.
 * @ingroup DeviceInterface
 */
DPCTL_API
bool DPCTLDevice_AreEq(__dpctl_keep const DPCTLSyclDeviceRef DRef1,
                       __dpctl_keep const DPCTLSyclDeviceRef DRef2);

/*!
 * @brief Checks if device has aspect.
 *
 * @param    DRef       Opaque pointer to a ``sycl::device``
 * @param    AT         DPCTLSyclAspectType of ``device::aspect``.
 * @return   True if sycl::device has device::aspect, else false.
 * @ingroup DeviceInterface
 */
DPCTL_API
bool DPCTLDevice_HasAspect(__dpctl_keep const DPCTLSyclDeviceRef DRef,
                           DPCTLSyclAspectType AT);

/*!
 * @brief Wrapper over
 * device.get_info<info::device::max_read_image_args>().
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   Returns the maximum number of simultaneous image objects that
 * can be read from by a kernel. The minimum value is 128 if the
 * SYCL device has aspect::image.
 */
DPCTL_API
uint32_t
DPCTLDevice_GetMaxReadImageArgs(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * device.get_info<info::device::max_write_image_args>().
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   Returns the maximum number of simultaneous image objects that
 *  can be written to by a kernel. The minimum value is 8 if the SYCL
 * device has aspect::image.
 */
DPCTL_API
uint32_t
DPCTLDevice_GetMaxWriteImageArgs(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * device.get_info<info::device::image2d_max_width>().
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   Returns the maximum width of a 2D image
 * or 1D image in pixels. The minimum value is
 * 8192 if the SYCL device has aspect::image.
 */
DPCTL_API
size_t
DPCTLDevice_GetImage2dMaxWidth(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * device.get_info<info::device::image2d_max_height>().
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   Returns the maximum height of a 2D image
 * or 1D image in pixels. The minimum value is
 * 8192 if the SYCL device has aspect::image.
 */
DPCTL_API
size_t
DPCTLDevice_GetImage2dMaxHeight(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * device.get_info<info::device::image3d_max_width>().
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   Returns the maximum width of a 3D image
 * in pixels. The minimum value is
 * 2048 if the SYCL device has aspect::image.
 */
DPCTL_API
size_t
DPCTLDevice_GetImage3dMaxWidth(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * device.get_info<info::device::image3d_max_height>().
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   Returns the maximum height of a 3D image
 * The minimum value is
 * 2048 if the SYCL device has aspect::image.
 */
DPCTL_API
size_t
DPCTLDevice_GetImage3dMaxHeight(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * device.get_info<info::device::image3d_max_depth>().
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   Returns the maximum depth of a 3D image
 * The minimum value is
 * 2048 if the SYCL device has aspect::image.
 */
DPCTL_API
size_t
DPCTLDevice_GetImage3dMaxDepth(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Returns a vector of sub devices
 * partitioned from this SYCL device based on the count parameter. The returned
 * vector contains as many sub devices as can be created such that each sub
 * device contains count compute units. If the device’s total number of compute
 * units is not evenly divided by count, then the remaining compute units are
 * not included in any of the sub devices.
 *
 * @param    DRef         Opaque pointer to a ``sycl::device``
 * @param    count        Count compute units that need to contains in
 * subdevices
 * @return   A #DPCTLDeviceVectorRef containing #DPCTLSyclDeviceRef objects
 * @ingroup DeviceInterface
 */
DPCTL_API
__dpctl_give DPCTLDeviceVectorRef
DPCTLDevice_CreateSubDevicesEqually(__dpctl_keep const DPCTLSyclDeviceRef DRef,
                                    size_t count);

/*!
 * @brief Returns a vector of sub devices
 * partitioned from this SYCL device based on the counts parameter. For each
 * non-zero value M in the counts vector, a sub device with M compute units
 * is created.
 *
 * @param    DRef         Opaque pointer to a ``sycl::device``
 * @param    counts       Array with count compute units
 * that need to contains in subdevices
 * @param    ncounts      Number of counts
 * @return   A #DPCTLDeviceVectorRef containing #DPCTLSyclDeviceRef objects
 * @ingroup DeviceInterface
 */
DPCTL_API
__dpctl_give DPCTLDeviceVectorRef
DPCTLDevice_CreateSubDevicesByCounts(__dpctl_keep const DPCTLSyclDeviceRef DRef,
                                     __dpctl_keep size_t *counts,
                                     size_t ncounts);

/*!
 * @brief Returns a vector of sub-devices
 * partitioned from this SYCL device by affinity domain based on the domain
 * parameter.
 *
 * @param    DRef         Opaque pointer to a ``sycl::device``
 * @param    PartAffDomTy A DPCTLPartitionAffinityDomainType enum value
 *
 * @return   A #DPCTLDeviceVectorRef containing #DPCTLSyclDeviceRef objects
 * @ingroup DeviceInterface
 */
DPCTL_API
__dpctl_give DPCTLDeviceVectorRef DPCTLDevice_CreateSubDevicesByAffinity(
    __dpctl_keep const DPCTLSyclDeviceRef DRef,
    DPCTLPartitionAffinityDomainType PartAffDomTy);
/*!
 * @brief Wrapper over
 * device.get_info<info::device::sub_group_independent_forward_progress>.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   Returns true if the device supports independent forward progress of
 * sub-groups with respect to other sub-groups in the same work-group.
 * @ingroup DeviceInterface
 */
DPCTL_API
bool DPCTLDevice_GetSubGroupIndependentForwardProgress(
    __dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * device.get_info<info::device::preferred_vector_width_char>.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   Returns the preferred native vector width size for built-in scalar
 * types that can be put into vectors.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint32_t DPCTLDevice_GetPreferredVectorWidthChar(
    __dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * device.get_info<info::device::preferred_vector_width_short>.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   Returns the preferred native vector width size for built-in scalar
 * types that can be put into vectors.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint32_t DPCTLDevice_GetPreferredVectorWidthShort(
    __dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * device.get_info<info::device::preferred_vector_width_int>.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   Returns the preferred native vector width size for built-in scalar
 * types that can be put into vectors.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint32_t DPCTLDevice_GetPreferredVectorWidthInt(
    __dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * device.get_info<info::device::preferred_vector_width_long>.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   Returns the preferred native vector width size for built-in scalar
 * types that can be put into vectors.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint32_t DPCTLDevice_GetPreferredVectorWidthLong(
    __dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * device.get_info<info::device::preferred_vector_width_float>.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   Returns the preferred native vector width size for built-in scalar
 * type.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint32_t DPCTLDevice_GetPreferredVectorWidthFloat(
    __dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * device.get_info<info::device::preferred_vector_width_double>.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   Returns the preferred native vector width size for built-in scalar
 * types that can be put into vectors.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint32_t DPCTLDevice_GetPreferredVectorWidthDouble(
    __dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * ``device.get_info<info::device::preferred_vector_width_half>``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   Returns the preferred native vector width size for built-in scalar
 * types that can be put into vectors.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint32_t DPCTLDevice_GetPreferredVectorWidthHalf(
    __dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * device.get_info<info::device::native_vector_width_char>.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   Returns the native ISA vector width size for built-in scalar
 * types that can be put into vectors.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint32_t DPCTLDevice_GetNativeVectorWidthChar(
    __dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * device.get_info<info::device::native_vector_width_short>.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   Returns the native ISA vector width size for built-in scalar
 * types that can be put into vectors.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint32_t DPCTLDevice_GetNativeVectorWidthShort(
    __dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * device.get_info<info::device::native_vector_width_int>.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   Returns the native ISA vector width size for built-in scalar
 * types that can be put into vectors.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint32_t
DPCTLDevice_GetNativeVectorWidthInt(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * device.get_info<info::device::native_vector_width_long>.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   Returns the native ISA vector width size for built-in scalar
 * types that can be put into vectors.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint32_t DPCTLDevice_GetNativeVectorWidthLong(
    __dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * device.get_info<info::device::native_vector_width_float>.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   Returns the native ISA vector width size for built-in scalar
 * type.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint32_t DPCTLDevice_GetNativeVectorWidthFloat(
    __dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * device.get_info<info::device::native_vector_width_double>.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   Returns the native ISA vector width size for built-in scalar
 * types that can be put into vectors.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint32_t DPCTLDevice_GetNativeVectorWidthDouble(
    __dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * ``device.get_info<info::device::native_vector_width_half>``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @return   Returns the native ISA vector width size for built-in scalar
 * types that can be put into vectors.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint32_t DPCTLDevice_GetNativeVectorWidthHalf(
    __dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * device.get_info<info::device::parent_device>
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   Returns an opaque pointer to the parent device for a sub-device,
 * or nullptr otherwise.
 */
DPCTL_API
__dpctl_give DPCTLSyclDeviceRef
DPCTLDevice_GetParentDevice(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * device.get_info<info::device::partition_max_sub_devices>
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   Returns the maximum number of sub-devices that can be created
 * when this device is partitioned.
 */
DPCTL_API
uint32_t DPCTLDevice_GetPartitionMaxSubDevices(
    __dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * std::hash<sycl::device>'s operator()
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   Returns hash value.
 */
DPCTL_API
size_t DPCTLDevice_Hash(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * device.get_info<info::device::profiling_timer_resolution>
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   Returns the resolution of device timer in nanoseconds.
 */
DPCTL_API
size_t DPCTLDevice_GetProfilingTimerResolution(
    __dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * device.get_info<info::device::global_mem_cache_line_size>
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   Returns the size of global memory cache line in bytes as uint32_t.
 */
DPCTL_API
uint32_t DPCTLDevice_GetGlobalMemCacheLineSize(
    __dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * device.get_info<info::device::max_clock_frequency>
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   Returns the maximum clock frequency in MHz as uint32_t.
 */
DPCTL_API
uint32_t
DPCTLDevice_GetMaxClockFrequency(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * device.get_info<info::device::max_mem_alloc_size>
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   Returns the maximum size of memory object in bytes as uint64_t.
 */
DPCTL_API
uint64_t
DPCTLDevice_GetMaxMemAllocSize(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * device.get_info<info::device::global_mem_cache_size>
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   Returns the size of global memory cache in bytes as uint64_t.
 */
DPCTL_API
uint64_t
DPCTLDevice_GetGlobalMemCacheSize(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper over
 * device.get_info<info::device::global_mem_cache_type>
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @return   Returns the type of global memory cache supported.
 */
DPCTL_API
DPCTLGlobalMemCacheType
DPCTLDevice_GetGlobalMemCacheType(__dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Wrapper for get_info<info::device::sub_group_sizes>().
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    res_len        Populated with size of the returned array
 * @return   Returns the valid result if device exists else returns NULL.
 * @ingroup DeviceInterface
 */
DPCTL_API
__dpctl_keep size_t *
DPCTLDevice_GetSubGroupSizes(__dpctl_keep const DPCTLSyclDeviceRef DRef,
                             size_t *res_len);

DPCTL_C_EXTERN_C_END
