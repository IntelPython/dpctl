//===-- dpctl_sycl_types.h - Defines opaque types for SYCL objects -*-C++-*- =//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2024 Intel Corporation
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
/// This file defines types used by dpctl's C interface to SYCL.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/ExternC.h"

DPCTL_C_EXTERN_C_BEGIN

/*!
 * @brief Opaque pointer to a ``sycl::context``
 *
 */
typedef struct DPCTLOpaqueSyclContext *DPCTLSyclContextRef;

/*!
 * @brief Opaque pointer to a ``sycl::device``
 *
 */
typedef struct DPCTLOpaqueSyclDevice *DPCTLSyclDeviceRef;

/*!
 * @brief Opaque pointer to a ``sycl::device_selector``
 *
 */
typedef struct DPCTLOpaqueSyclDeviceSelector *DPCTLSyclDeviceSelectorRef;

/*!
 * @brief Opaque pointer to a ``sycl::event``
 *
 */
typedef struct DPCTLOpaqueSyclEvent *DPCTLSyclEventRef;

/*!
 * @brief Opaque pointer to a ``sycl::kernel``
 *
 */
typedef struct DPCTLOpaqueSyclKernel *DPCTLSyclKernelRef;

/*!
 * @brief Opaque pointer to a
 * ``sycl::kernel_bundle<sycl::bundle_state::executable>``
 *
 */
typedef struct DPCTLOpaqueSyclKernelBundle *DPCTLSyclKernelBundleRef;

/*!
 * @brief Opaque pointer to a ``sycl::platform``
 *
 */
typedef struct DPCTLOpaqueSyclPlatform *DPCTLSyclPlatformRef;

/*!
 * @brief Opaque pointer to a ``sycl::queue``
 *
 */
typedef struct DPCTLOpaqueSyclQueue *DPCTLSyclQueueRef;

/*!
 * @brief Used to pass a ``sycl::usm`` memory opaquely through DPCTL interfaces.
 *
 */
typedef struct DPCTLOpaqueSyclUSM *DPCTLSyclUSMRef;

DPCTL_C_EXTERN_C_END
