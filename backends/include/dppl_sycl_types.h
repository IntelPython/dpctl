//===---------- dppl_sycl_types.h - DPPL-SYCL interface ---*--- C++ ---*---===//
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
/// This file defines opaque pointer types wrapping Sycl object that get used
/// by DPPL's C API.
///
//===----------------------------------------------------------------------===//

#pragma once

/*!
 * @brief Opaque pointer used to represent references to sycl::context*
 *
 * @see sycl::context
 */
typedef struct DPPLOpaqueSyclContext *DPPLSyclContextRef;

/*!
 * @brief Opaque pointer used to represent references to sycl::device*
 *
 * @see sycl::device
 */
typedef struct DPPLOpaqueSyclDevice *DPPLSyclDeviceRef;

/*!
 * @brief Opaque pointer used to represent references to sycl::event*
 *
 * @see sycl::event
 */
typedef struct DPPLOpaqueSyclEvent *DPPLSyclEventRef;

/*!
 * @brief Opaque pointer used to represent references to sycl::kernel*
 *
 * @see sycl::kernel
 */
typedef struct DPPLOpaqueSyclProgram *DPPLSyclKernelRef;

/*!
 * @brief Opaque pointer used to represent references to sycl::platform*
 *
 * @see sycl::platform
 */
typedef struct DPPLOpaqueSyclPlatform *DPPLSyclPlatformRef;

/*!
 * @brief Opaque pointer used to represent references to sycl::program*
 *
 * @see sycl::program
 */
typedef struct DPPLOpaqueSyclProgram *DPPLSyclProgramRef;

 /*!
  * @brief Opaque pointer used to represent references to sycl::queue*
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
