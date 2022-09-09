//===-- dpctl_utils.h - Enum to string helper functions          -*-C++-*- ===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2022 Intel Corporation
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
/// This file defines common helper functions used in other places in dpctl.
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/DllExport.h"
#include "dpctl_sycl_enum_types.h"
#include <CL/sycl.hpp>

/*!
 * @brief Converts a sycl::info::device_type input value to a string.
 *
 * @param    devTy          A sycl::info::device_type enum value.
 * @return   A string representation of a sycl::info::device_type enum.
 */
DPCTL_API
std::string DPCTL_DeviceTypeToStr(sycl::info::device_type devTy);

/*!
 * @brief Converts a string to sycl::info::device_type enum value.
 *
 * Tries to interpret the input string a return a corresponding device_type. If
 * no conversion is possible, then a runtime_error is thrown.
 *
 * @param    devTyStr       Input string for which we search a
 *                          sycl::info::device_type enum value.
 * @return   The sycl::info::device_type enum value corresponding to the input
 * string.
 * @throws runtime_error
 */
DPCTL_API
sycl::info::device_type DPCTL_StrToDeviceType(const std::string &devTyStr);

/*!
 * @brief Converts a DPCTLSyclBackendType enum value to its corresponding
 * sycl::backend enum value. If conversion fails, a runtime_error is thrown.
 *
 * @param    BeTy           My Param doc
 * @return   A sycl::backend enum value for the input
 * DPCTLSyclDeviceType enum value.
 * @throws runtime_error
 */
DPCTL_API
sycl::backend DPCTL_DPCTLBackendTypeToSyclBackend(DPCTLSyclBackendType BeTy);

/*!
 * @brief Converts a sycl::backend enum value to corresponding
 * DPCTLSyclBackendType enum value.
 *
 * @param    B           sycl::backend to be converted to
 *                       DPCTLSyclBackendType enum.
 * @return   A DPCTLSyclBackendType enum value for the input
 * sycl::backend enum value.
 */
DPCTL_API
DPCTLSyclBackendType DPCTL_SyclBackendToDPCTLBackendType(sycl::backend B);

/*!
 * @brief Converts a DPCTLSyclDeviceType enum value to its corresponding
 * sycl::info::device_type enum value. If conversion fails, a runtime_error is
 * thrown.
 *
 * @param    DTy           A DPCTLSyclDeviceType enum value
 * @return   A sycl::info::device_type enum value for the input
 * DPCTLSyclDeviceType enum value.
 * @throws runtime_error
 */
DPCTL_API
sycl::info::device_type
DPCTL_DPCTLDeviceTypeToSyclDeviceType(DPCTLSyclDeviceType DTy);

/*!
 * @brief Converts a sycl::info::device_type enum value to corresponding
 * DPCTLSyclDeviceType enum value.
 *
 * @param    D           sycl::info::device_type to be converted to
 *                       DPCTLSyclDeviceType enum.
 * @return   A DPCTLSyclDeviceType enum value for the input
 * sycl::info::device_type enum value.
 */
DPCTL_API
DPCTLSyclDeviceType
DPCTL_SyclDeviceTypeToDPCTLDeviceType(sycl::info::device_type D);

/*!
 * @brief Converts a sycl::aspect input value to a string.
 *
 * @param    aspectTy          A sycl::aspect value.
 * @return   A string representation of a sycl::aspect.
 * @throws runtime_error
 */
DPCTL_API
std::string DPCTL_AspectToStr(sycl::aspect aspectTy);

/*!
 * @brief Converts a string to sycl::aspect value.
 *
 * @param    aspectTyStr       Input string for which we search a
 *                             sycl::aspect value.
 * @return   The sycl::aspect value corresponding to the input
 * string.
 * @throws runtime_error
 */
DPCTL_API
sycl::aspect DPCTL_StrToAspectType(const std::string &aspectTyStr);

/*!
 * @brief Converts a DPCTLSyclAspectType enum value to its corresponding
 * sycl::aspect enum value.
 *
 * @param    AspectTy           A DPCTLSyclAspectType enum value
 * @return   A sycl::aspect enum value for the input
 * DPCTLSyclAspectType enum value.
 * @throws runtime_error
 */
DPCTL_API
sycl::aspect DPCTL_DPCTLAspectTypeToSyclAspect(DPCTLSyclAspectType AspectTy);

/*!
 * @brief Converts a sycl::aspect enum value to corresponding
 * DPCTLSyclAspectType enum value.
 *
 * @param    Aspect           sycl::aspect to be converted to
 *                            DPCTLSyclAspectType enum.
 * @return   A DPCTLSyclAspectType enum value for the input
 * sycl::aspect enum value.
 * @throws runtime_error
 */
DPCTL_API
DPCTLSyclAspectType DPCTL_SyclAspectToDPCTLAspectType(sycl::aspect Aspect);

/*!
 * @brief Converts a DPCTLPartitionAffinityDomainType enum value to its
 * corresponding sycl::info::partition_affinity_domain enum value.
 *
 * @param    PartitionAffinityDomainTy           A
 * DPCTLPartitionAffinityDomainType enum value
 * @return   A sycl::info::partition_affinity_domain enum value for the input
 * DPCTLPartitionAffinityDomainType enum value.
 * @throws runtime_error
 */
DPCTL_API
sycl::info::partition_affinity_domain
DPCTL_DPCTLPartitionAffinityDomainTypeToSycl(
    DPCTLPartitionAffinityDomainType PartitionAffinityDomainTy);

/*!
 * @brief Converts a sycl::info::partition_affinity_domain enum value to
 * corresponding DPCTLPartitionAffinityDomainType enum value.
 *
 * @param    PartitionAffinityDomain sycl::info::partition_affinity_domain to be
 * converted to DPCTLPartitionAffinityDomainType enum.
 * @return   A DPCTLPartitionAffinityDomainType enum value for the input
 * sycl::info::partition_affinity_domain enum value.
 * @throws runtime_error
 */
DPCTL_API
DPCTLPartitionAffinityDomainType DPCTL_SyclPartitionAffinityDomainToDPCTLType(
    sycl::info::partition_affinity_domain PartitionAffinityDomain);

/*!
 * @brief Gives the index of the given device with respective to all the other
 * devices of the same type in the device's platform.
 *
 * The relative device id of a device (Device) is computed by looking up the
 * position of Device in the ``std::vector`` returned by calling the
 * ``get_devices(Device.get_info<sycl::info::device::device_type>())`` function
 * for Device's platform. A relative device id of -1 indicates that the relative
 * id could not be computed.
 *
 * @param    Device         A ``sycl::device`` object whose relative id is to be
 *                          computed.
 * @return   A relative id corresponding to the device, -1 indicates that a
 * relative id value could not be computed.
 */
DPCTL_API
int64_t DPCTL_GetRelativeDeviceId(const sycl::device &Device);

/*!
 * @brief Gives the filter string which would select given root device if
 * used as argument to ``sycl::ext::oneapi::filter_selector``. Throws exception
 * if filter string can not be constructed.
 *
 * @param    Device         A ``sycl::device`` object whose filter selector
 *                          needs to be computed.
 * @return   Filter selector for the device.
 */
DPCTL_API
std::string DPCTL_GetDeviceFilterString(const sycl::device &Device);

/*!
 * @brief Converts a ``sycl::info::event_command_status`` enum value to
 * corresponding DPCTLSyclEventStatusType enum value.
 *
 * @param    SyclESTy           ``sycl::info::event_command_status`` to be
 * converted to DPCTLSyclEventStatusType enum.
 * @return   A DPCTLSyclEventStatusType enum value for the input
 * ``sycl::info::event_command_status`` enum value.
 */
DPCTL_API
DPCTLSyclEventStatusType DPCTL_SyclEventStatusToDPCTLEventStatusType(
    sycl::info::event_command_status SyclESTy);
