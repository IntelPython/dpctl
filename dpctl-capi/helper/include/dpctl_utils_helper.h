//===-- dpctl_utils.h - Enum to string helper functions          -*-C++-*- ===//
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
/// This file defines common helper functions used in other places in dpCtl.
//===----------------------------------------------------------------------===//

#pragma once

#include "../include/dpctl_sycl_enum_types.h"
#include <CL/sycl.hpp>

/*!
 * @brief Converts a sycl::info::device_type input value to a string.
 *
 * @param    devTy          A sycl::info::device_type enum value.
 * @return   A string representation of a sycl::info::device_type enum.
 */
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
DPCTLSyclDeviceType
DPCTL_SyclDeviceTypeToDPCTLDeviceType(sycl::info::device_type D);

/*!
 * @brief Converts a sycl::aspect input value to a string.
 *
 * @param    aspectTy          A sycl::aspect value.
 * @return   A string representation of a sycl::aspect.
 * @throws runtime_error
 */
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
DPCTLSyclAspectType DPCTL_SyclAspectToDPCTLAspectType(sycl::aspect Aspect);
