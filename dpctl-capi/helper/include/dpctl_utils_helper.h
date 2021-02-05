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

#include <CL/sycl.hpp>
#include "../include/dpctl_sycl_enum_types.h"

using namespace cl::sycl;

/*!
 * @brief
 *
 * @param    devTy          My Param doc
 * @return   {return}       My Param doc
 */
std::string DPCTL_DeviceTypeToStr (info::device_type devTy);

/*!
 * @brief
 *
 * @param    devTyStr       My Param doc
 * @return   {return}       My Param doc
 */
info::device_type DPCTL_StrToDeviceType (const std::string & devTyStr);

/*!
 * @brief
 *
 * @param    BeTy           My Param doc
 * @return   {return}       My Param doc
 */
backend DPCTL_DPCTLBackendTypeToSyclBackend (const DPCTLSyclBackendType & BeTy);

/*!
 * @brief
 *
 * @param    BeTy           My Param doc
 * @return   {return}       My Param doc
 */
info::device_type DPCTL_DPCTLDeviceTypeToSyclDeviceType (
    const DPCTLSyclDeviceType & DTy);
