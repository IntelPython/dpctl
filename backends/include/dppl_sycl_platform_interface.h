//===--- dppl_sycl_platform_interface.h - DPPL-SYCL interface ---*--C++ -*-===//
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
/// This header declares a C interface to sycl::platform interface functions.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "dppl_data_types.h"

DPPL_C_EXTERN_C_BEGIN

/*!
 * @brief Get the number of sycl::platform available on the system.
 *
 * @return The number of available sycl::platforms.
 */
DPPL_API
size_t DPPLPlatform_GetNumPlatforms();

/*!
 * @brief Prints out some selected info about all sycl::platform on the system.
 *
 */
DPPL_API
void DPPLPlatform_DumpInfo();

DPPL_C_EXTERN_C_END
