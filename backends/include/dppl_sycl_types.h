//===-------------- dppl_sycl_types.h - dpctl-C_API ----*---- C++ ----*----===//
//
//               Data Parallel Control Library (dpCtl)
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
/// This file defines types used by DPPL's C interface to SYCL.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/ExternC.h"

DPPL_C_EXTERN_C_BEGIN

/*!
 * @brief Opaque pointer to a sycl::context
 *
 */
typedef struct DPPLOpaqueSyclContext *DPPLSyclContextRef;

/*!
 * @brief Opaque pointer to a sycl::device
 *
 */
typedef struct DPPLOpaqueSyclDevice *DPPLSyclDeviceRef;

/*!
 * @brief Opaque pointer to a sycl::event
 *
 */
typedef struct DPPLOpaqueSyclEvent *DPPLSyclEventRef;

/*!
 * @brief Opaque pointer to a sycl::kernel
 *
 */
typedef struct DPPLOpaqueSyclKernel *DPPLSyclKernelRef;

/*!
 * @brief Opaque pointer to a sycl::platform
 *
 */
typedef struct DPPLOpaqueSyclPlatform *DPPLSyclPlatformRef;

/*!
 * @brief Opaque pointer to a sycl::program
 *
 */
typedef struct DPPLOpaqueSyclProgram *DPPLSyclProgramRef;

 /*!
  * @brief Opaque pointer to a sycl::queue
  *
  * @see sycl::queue
  */
typedef struct DPPLOpaqueSyclQueue *DPPLSyclQueueRef;

/*!
 * @brief Used to pass a sycl::usm memory opaquely through DPPL interfaces.
 *
 * @see sycl::usm
 */
typedef struct DPPLOpaqueSyclUSM *DPPLSyclUSMRef;

DPPL_C_EXTERN_C_END
