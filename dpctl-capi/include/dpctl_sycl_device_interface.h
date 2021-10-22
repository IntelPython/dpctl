//===----- dpctl_sycl_device_interface.h - C API for sycl::device -*-C++-*- ==//
//
//                      Data Parallel Control (dpctl)
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

#include "Support/MemOwnershipAttrs.h"
#include "Support/elementary_macros.h"
#include "dpctl_data_types.h"
#include "dpctl_exec_state.h"
#include "dpctl_sycl_device_manager.h"
#include "dpctl_sycl_enum_types.h"
#include "dpctl_sycl_types.h"

DPCTL_C_EXTERN_C_BEGIN

/**
 * @defgroup DeviceInterface Device class C wrapper
 */

/*!
 * @brief Returns a copy of the ``DpctlSyclDeviceRef`` object.
 *
 * @param    DRef           ``DpctlSyclDeviceRef`` object to be copied.
 * @param    ES             The execution state object used for error handling.
 * @return   A new ``DpctlSyclDeviceRef`` created by copying the passed in
 * ``DpctlSyclDeviceRef`` object.
 * @ingroup DeviceInterface
 */
DPCTL_API
__dpctl_give DpctlSyclDeviceRef
dpctl_device_copy(__dpctl_keep const DpctlSyclDeviceRef DRef,
                  __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Returns a new ``DpctlSyclDeviceRef`` opaque object wrapping a SYCL
 * device instance as a host device.
 *
 * @param    ES             The execution state object used for error handling.
 * @return   An opaque pointer to a ``sycl::device`` created as an instance
 * of the host device.
 * @ingroup DeviceInterface
 */
DPCTL_API __dpctl_give DpctlSyclDeviceRef
dpctl_device_create(__dpctl_keep const DpctlExecState ES);

/*!
 * @brief Returns a new ``DpctlSyclDeviceRef`` opaque object created using the
 * provided device_selector.
 *
 * @param    DSRef          An opaque pointer to a ``sycl::device_selector``.
 * @param    ES             The execution state object used for error handling.
 * @return   Returns an opaque pointer to a SYCL device created using the
 *           device_selector, if the requested device could not be created a
 *           nullptr is returned.
 * @ingroup DeviceInterface
 */
DPCTL_API
__dpctl_give DpctlSyclDeviceRef dpctl_device_create_from_selector(
    __dpctl_keep const DPCTLSyclDeviceSelectorRef DSRef,
    __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Deletes a ``DpctlSyclDeviceRef`` pointer after casting to to
 * ``sycl::device``.
 *
 * @param    DRef           The ``DpctlSyclDeviceRef`` pointer to be freed.
 * @param    ES             The execution state object used for error handling.
 * @ingroup DeviceInterface
 */
DPCTL_API
void dpctl_device_delete(__dpctl_take DpctlSyclDeviceRef DRef,
                         __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Returns true if the device type is
 * ``sycl::info::device_type::accelerator``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   True if the device type is an accelerator, else False.
 * @ingroup DeviceInterface
 */
DPCTL_API
bool dpctl_device_is_accelerator(__dpctl_keep const DpctlSyclDeviceRef DRef,
                                 __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Returns true if the device type is ``sycl::info::device_type::cpu``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   True if the device type is a cpu, else False.
 * @ingroup DeviceInterface
 */
DPCTL_API
bool dpctl_device_is_cpu(__dpctl_keep const DpctlSyclDeviceRef DRef,
                         __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Returns true if the device type is ``sycl::info::device_type::gpu``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return    True if the device type is a gpu, else False.
 * @ingroup DeviceInterface
 */
DPCTL_API
bool dpctl_device_is_gpu(__dpctl_keep const DpctlSyclDeviceRef DRef,
                         __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Returns true if this SYCL device is a host device.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   True if the device is a host device, else False.
 * @ingroup DeviceInterface
 */
DPCTL_API
bool dpctl_device_is_host(__dpctl_keep const DpctlSyclDeviceRef DRef,
                          __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Returns the backend for the device.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   A ``DPCTLSyclBackendType`` enum value representing the
 * ``sycl::backend`` for the device.
 * @ingroup DeviceInterface
 */
DPCTL_API
DPCTLSyclBackendType
dpctl_device_get_backend(__dpctl_keep const DpctlSyclDeviceRef DRef,
                         __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Returns the ``DPCTLSyclDeviceType`` enum value for the
 * ``DpctlSyclDeviceRef`` argument.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   The ``DPCTLSyclDeviceType`` value corresponding to the device.
 * @ingroup DeviceInterface
 */
DPCTL_API
DPCTLSyclDeviceType
dpctl_device_get_device_type(__dpctl_keep const DpctlSyclDeviceRef DRef,
                             __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Returns the driver version as a C string.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   A C string in the form major_number.minor.number that corresponds
 *           to the driver version.
 * @ingroup DeviceInterface
 */
DPCTL_API
__dpctl_give const char *
dpctl_device_get_driver_version(__dpctl_keep const DpctlSyclDeviceRef DRef,
                                __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Wrapper over ``device.get_info<info::device::max_compute_units>()``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   Returns the valid result if device exists else returns 0.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint32_t
dpctl_device_get_max_compute_units(__dpctl_keep const DpctlSyclDeviceRef DRef,
                                   __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Wrapper over ``device.get_info<info::device::global_mem_size>()``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   Returns the valid result if device exists else returns 0.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint64_t
dpctl_device_get_global_mem_size(__dpctl_keep const DpctlSyclDeviceRef DRef,
                                 __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Wrapper over ``device.get_info<info::device::local_mem_size>()``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   Returns the valid result if device exists else returns 0.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint64_t
dpctl_device_get_local_mem_size(__dpctl_keep const DpctlSyclDeviceRef DRef,
                                __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Wrapper for ``get_info<info::device::max_work_item_dimensions>()``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   Returns the valid result if device exists else returns 0.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint32_t
dpctl_device_get_max_work_item_dims(__dpctl_keep const DpctlSyclDeviceRef DRef,
                                    __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Wrapper for ``get_info<info::device::max_work_item_sizes>()``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   Returns the valid result if device exists else returns NULL.
 * @ingroup DeviceInterface
 */
DPCTL_API
__dpctl_keep size_t *
dpctl_device_get_max_work_item_sizes(__dpctl_keep const DpctlSyclDeviceRef DRef,
                                     __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Wrapper for ``get_info<info::device::max_work_group_size>()``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   Returns the valid result if device exists else returns 0.
 * @ingroup DeviceInterface
 */
DPCTL_API
size_t
dpctl_device_get_max_work_group_size(__dpctl_keep const DpctlSyclDeviceRef DRef,
                                     __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Wrapper over ``device.get_info<info::device::max_num_sub_groups>()``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   Returns the valid result if device exists else returns 0.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint32_t
dpctl_device_get_max_num_sub_groups(__dpctl_keep const DpctlSyclDeviceRef DRef,
                                    __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Returns the ``sycl::platform`` for the device as a
 * ``DpctlSyclPlatformRef`` opaque pointer.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   An opaque pointer to the ``sycl::platform`` for the device.
 * @ingroup DeviceInterface
 */
DPCTL_API
__dpctl_give DpctlSyclPlatformRef
dpctl_device_get_platform(__dpctl_keep const DpctlSyclDeviceRef DRef,
                          __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Returns a C string for the device name.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   A C string containing the device name.
 * @ingroup DeviceInterface
 */
DPCTL_API
__dpctl_give const char *
dpctl_device_get_name(__dpctl_keep const DpctlSyclDeviceRef DRef,
                      __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Returns a C string corresponding to the devices's vendor name.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return    A C string containing the vendor name.
 * @ingroup DeviceInterface
 */
DPCTL_API
__dpctl_give const char *
dpctl_device_get_vendor(__dpctl_keep const DpctlSyclDeviceRef DRef,
                        __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Returns true if the device and the host share a unified memory
 * subsystem, else returns false.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   Boolean indicating if the device shares a unified memory subsystem
 * with the host.
 * @ingroup DeviceInterface
 */
DPCTL_API
bool dpctl_device_is_host_unified_memory(__dpctl_keep const DpctlSyclDeviceRef
                                             DRef,
                                         __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Checks if two ``DpctlSyclDeviceRef`` objects point to the same
 * ``sycl::device``.
 *
 * @param    DRef1         First opaque pointer to a ``sycl::device``.
 * @param    DRef2         Second opaque pointer to a ``sycl::device``.
 * @param    ES             The execution state object used for error handling.
 * @return   True if the underlying sycl::device are same, false otherwise.
 * @ingroup DeviceInterface
 */
DPCTL_API
bool dpctl_device_are_eq(__dpctl_keep const DpctlSyclDeviceRef DRef1,
                         __dpctl_keep const DpctlSyclDeviceRef DRef2,
                         __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Checks if device supports the specified aspect.
 *
 * @param    DRef       Opaque pointer to a ``sycl::device``
 * @param    AT         DPCTLSyclAspectType of ``device::aspect``.
 * @param    ES         The execution state object used for error handling.
 * @return   True if ``sycl::device`` has ``device::aspect``, else false.
 * @ingroup DeviceInterface
 */
DPCTL_API
bool dpctl_device_has_aspect(__dpctl_keep const DpctlSyclDeviceRef DRef,
                             DPCTLSyclAspectType AT,
                             __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Wrapper over
 * ``device.get_info<info::device::max_read_image_args>()``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   Returns the maximum number of simultaneous image objects that
 * can be read from by a kernel. The minimum value is 128 if the
 * SYCL device has aspect::image.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint32_t
dpctl_device_get_max_read_image_args(__dpctl_keep const DpctlSyclDeviceRef DRef,
                                     __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Wrapper over
 * ``device.get_info<info::device::max_write_image_args>()``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   Returns the maximum number of simultaneous image objects that
 *  can be written to by a kernel. The minimum value is 8 if the SYCL
 * device has ``aspect::image``.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint32_t dpctl_device_get_max_write_image_args(
    __dpctl_keep const DpctlSyclDeviceRef DRef,
    __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Wrapper over ``device.get_info<info::device::image2d_max_width>()``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   Returns the maximum width of a 2D image  or 1D image in pixels. The
 * minimum value is 8192 if the SYCL device has ``aspect::image``.
 * @ingroup DeviceInterface
 */
DPCTL_API
size_t
dpctl_device_get_image2d_max_width(__dpctl_keep const DpctlSyclDeviceRef DRef,
                                   __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Wrapper over ``device.get_info<info::device::image2d_max_height>()``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   Returns the maximum height of a 2D image or 1D image in pixels. The
 * minimum value is 8192 if the SYCL device has ``aspect::image``.
 * @ingroup DeviceInterface
 */
DPCTL_API
size_t
dpctl_device_get_image2d_max_height(__dpctl_keep const DpctlSyclDeviceRef DRef,
                                    __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Wrapper over ``device.get_info<info::device::image3d_max_width>()``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   Returns the maximum width of a 3D image in pixels. The minimum
 * value is 2048 if the SYCL device has ``aspect::image``.
 * @ingroup DeviceInterface
 */
DPCTL_API
size_t
dpctl_device_get_image3d_max_width(__dpctl_keep const DpctlSyclDeviceRef DRef,
                                   __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Wrapper over ``device.get_info<info::device::image3d_max_height>()``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   Returns the maximum height of a 3D image. The minimum value is
 * 2048 if the SYCL device has ``aspect::image``.
 * @ingroup DeviceInterface
 */
DPCTL_API
size_t
dpctl_device_get_image3d_max_height(__dpctl_keep const DpctlSyclDeviceRef DRef,
                                    __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Wrapper over ``device.get_info<info::device::image3d_max_depth>()``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   Returns the maximum depth of a 3D image. The minimum value is 2048
 * if the SYCL device has ``aspect::image``.
 * @ingroup DeviceInterface
 */
DPCTL_API
size_t
dpctl_device_get_image3d_max_depth(__dpctl_keep const DpctlSyclDeviceRef DRef,
                                   __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Returns a vector of sub-devices partitioned from this SYCL device
 * based on the count parameter.
 *
 * The returned  vector contains as many sub-devices as can be created such that
 * each sub device contains count compute units. If the device’s total number of
 * compute units is not evenly divided by count, then the remaining compute
 * units are not included in any of the sub devices.
 *
 * @param    DRef       Opaque pointer to a ``sycl::device``.
 * @param    count      Count compute units that need to contains in subdevices.
 * @param    ES         The execution state object used for error handling.
 * @return   A #DPCTLDeviceVectorRef containing #DpctlSyclDeviceRef objects
 * @ingroup DeviceInterface
 */
DPCTL_API
__dpctl_give DPCTLDeviceVectorRef dpctl_device_create_sub_devices_equally(
    __dpctl_keep const DpctlSyclDeviceRef DRef,
    size_t count,
    __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Returns a vector of sub-devices partitioned from this SYCL device
 * based on the counts parameter.
 *
 * For each non-zero value M in the counts vector, a sub device with M compute
 * units is created.
 *
 * @param    DRef         Opaque pointer to a ``sycl::device``
 * @param    counts       Array with count compute units
 *                        that need to contains in subdevices
 * @param    ncounts      Number of counts
 * @param    ES           The execution state object used for error handling.
 * @return   A #DPCTLDeviceVectorRef containing #DpctlSyclDeviceRef objects
 * @ingroup DeviceInterface
 */
DPCTL_API
__dpctl_give DPCTLDeviceVectorRef dpctl_device_create_sub_devices_by_counts(
    __dpctl_keep const DpctlSyclDeviceRef DRef,
    __dpctl_keep size_t *counts,
    size_t ncounts,
    __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Returns a vector of sub-devices partitioned from this SYCL device by
 * affinity domain based on the domain parameter.
 *
 * @param    DRef         Opaque pointer to a ``sycl::device``
 * @param    PartAffDomTy A DPCTLPartitionAffinityDomainType enum value
 * @param    ES           The execution state object used for error handling.
 * @return   A #DPCTLDeviceVectorRef containing #DpctlSyclDeviceRef objects
 * @ingroup DeviceInterface
 */
DPCTL_API
__dpctl_give DPCTLDeviceVectorRef dpctl_device_create_sub_devices_by_affinity(
    __dpctl_keep const DpctlSyclDeviceRef DRef,
    DPCTLPartitionAffinityDomainType PartAffDomTy,
    __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Wrapper over
 *  ``device.get_info<info::device::sub_group_independent_forward_progress>()``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   Returns true if the device supports independent forward progress of
 * sub-groups with respect to other sub-groups in the same work-group.
 * @ingroup DeviceInterface
 */
DPCTL_API
bool dpctl_device_get_sub_group_independent_forward_progress(
    __dpctl_keep const DpctlSyclDeviceRef DRef,
    __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Wrapper over
 * ``device.get_info<info::device::preferred_vector_width_char>()``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   Returns the preferred native vector width size for built-in scalar
 * types that can be put into vectors.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint32_t dpctl_device_get_preferred_vector_width_char(
    __dpctl_keep const DpctlSyclDeviceRef DRef,
    __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Wrapper over
 * ``device.get_info<info::device::preferred_vector_width_short>()``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   Returns the preferred native vector width size for built-in scalar
 * types that can be put into vectors.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint32_t dpctl_device_get_preferred_vector_width_short(
    __dpctl_keep const DpctlSyclDeviceRef DRef,
    __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Wrapper over
 * ``device.get_info<info::device::preferred_vector_width_int>()``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   Returns the preferred native vector width size for built-in scalar
 * types that can be put into vectors.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint32_t dpctl_device_get_preferred_vector_width_int(
    __dpctl_keep const DpctlSyclDeviceRef DRef,
    __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Wrapper over
 * ``device.get_info<info::device::preferred_vector_width_long>()``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   Returns the preferred native vector width size for built-in scalar
 * types that can be put into vectors.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint32_t dpctl_device_get_preferred_vector_width_long(
    __dpctl_keep const DpctlSyclDeviceRef DRef,
    __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Wrapper over
 * ``device.get_info<info::device::preferred_vector_width_float>()``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   Returns the preferred native vector width size for built-in scalar
 * type.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint32_t dpctl_device_get_preferred_vector_width_float(
    __dpctl_keep const DpctlSyclDeviceRef DRef,
    __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Wrapper over
 * ``device.get_info<info::device::preferred_vector_width_double>``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   Returns the preferred native vector width size for built-in scalar
 * types that can be put into vectors.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint32_t dpctl_device_get_preferred_vector_width_double(
    __dpctl_keep const DpctlSyclDeviceRef DRef,
    __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Wrapper over
 * ``device.get_info<info::device::preferred_vector_width_half>``.
 *
 * @param    DRef           Opaque pointer to a ``sycl::device``
 * @param    ES             The execution state object used for error handling.
 * @return   Returns the preferred native vector width size for built-in scalar
 * types that can be put into vectors.
 * @ingroup DeviceInterface
 */
DPCTL_API
uint32_t dpctl_device_get_preferred_vector_width_half(
    __dpctl_keep const DpctlSyclDeviceRef DRef,
    __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Wrapper over ``device.get_info<info::device::parent_device>``
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @param    ES             The execution state object used for error handling.
 * @return   Returns an opaque pointer to the parent device for a sub-device,
 * or nullptr otherwise.
 */
DPCTL_API
__dpctl_give DpctlSyclDeviceRef
dpctl_device_get_parent_device(__dpctl_keep const DpctlSyclDeviceRef DRef,
                               __dpctl_keep const DpctlExecState ES);

/*!
 * @brief Wrapper over ``std::hash<sycl::device>``'s ``operator()``
 *
 * @param    DRef           Opaque pointer to a sycl::device
 * @param    ES             The execution state object used for error handling.
 * @return   Returns hash value.
 */
DPCTL_API
size_t dpctl_device_hash(__dpctl_keep const DpctlSyclDeviceRef DRef,
                         __dpctl_keep const DpctlExecState ES);

DPCTL_C_EXTERN_C_END
