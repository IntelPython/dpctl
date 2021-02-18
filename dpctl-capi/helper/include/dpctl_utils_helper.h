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
 * @brief
 *
 * @param    devTy          My Param doc
 * @return   {return}       My Param doc
 */
std::string DPCTL_DeviceTypeToStr(sycl::info::device_type devTy);

/*!
 * @brief
 *
 * @param    devTyStr       My Param doc
 * @return   {return}       My Param doc
 */
sycl::info::device_type DPCTL_StrToDeviceType(const std::string &devTyStr);

/*!
 * @brief
 *
 * @param    BeTy           My Param doc
 * @return   {return}       My Param doc
 */
sycl::backend DPCTL_DPCTLBackendTypeToSyclBackend(DPCTLSyclBackendType BeTy);

/*!
 * @brief
 *
 * @param    B           My Param doc
 * @return   {return}    My Param doc
 */
DPCTLSyclBackendType DPCTL_SyclBackendToDPCTLBackendType(sycl::backend B);

/*!
 * @brief
 *
 * @param    DTy           My Param doc
 * @return   {return}      My Param doc
 */
sycl::info::device_type
DPCTL_DPCTLDeviceTypeToSyclDeviceType(DPCTLSyclDeviceType DTy);

/*!
 * @brief
 *
 * @param    D           My Param doc
 * @return   {return}    My Param doc
 */
DPCTLSyclDeviceType
DPCTL_SyclDeviceTypeToDPCTLDeviceType(sycl::info::device_type D);
