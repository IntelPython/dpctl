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
/// This file defines types used by DPPL's C interface to SYCL.
///
//===----------------------------------------------------------------------===//

#pragma once

/*!
 * @brief
 *
 */
typedef struct DPPLOpaqueSyclContext *DPPLSyclContextRef;

/*!
 * @brief
 *
 */
typedef struct DPPLOpaqueSyclDevice *DPPLSyclDeviceRef;

/*!
 * @brief
 *
 */
typedef struct DPPLOpaqueSyclPlatform *DPPLSyclPlatformRef;

 /*!
  * @brief Used to pass a sycl::queue opaquely through DPPL interfaces.
  *
  * @see sycl::queue
  */
typedef struct DPPLOpaqueSyclQueue *DPPLSyclQueueRef;

/*!
 * @brief Used to pass a sycl::program opaquely through DPPL interfaces.
 *
 */
typedef struct DPPLOpaqueSyclProgram *DPPLSyclProgramRef;

/*!
 * @brief Used to pass a sycl::usm memory opaquely through DPPL interfaces.
 *
 * @see sycl::usm
 */
typedef struct DPPLOpaqueSyclUSM *DPPLSyclUSMRef;
