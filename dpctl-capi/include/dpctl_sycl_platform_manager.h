//===-- dpctl_sycl_platform_manager.h - Helpers for sycl::platform -*-C++-*- =//
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
/// This header declares helper functions to work with  sycl::platform objects.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"
#include "dpctl_sycl_types.h"
#include "dpctl_vector.h"

DPCTL_C_EXTERN_C_BEGIN

/*! \addtogroup PlatformManager Platform class helper functions
 * Helper functions for ``sycl::platform`` objects that do not directly map to
 * any ``sycl::platform`` member function.
 * @{
 */

// Declares a set of types abd functions to deal with vectors of
// DPCTLSyclPlatformRef. Refer dpctl_vector_macros.h
DPCTL_DECLARE_VECTOR(Platform)

/*!
 * @brief Prints out information about the sycl::platform argument.
 * @param    PRef           A #DPCTLSyclPlatformRef opaque pointer.
 */
DPCTL_API
void DPCTLPlatformMgr_PrintInfo(__dpctl_keep const DPCTLSyclPlatformRef PRef);

/*! @} */

DPCTL_C_EXTERN_C_END
