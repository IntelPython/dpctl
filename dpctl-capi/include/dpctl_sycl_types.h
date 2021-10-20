//===-- dpctl_sycl_types.h - Defines opaque types for SYCL objects -*-C++-*- =//
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
/// This file defines types used by dpctl's C interface to SYCL.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/ExternC.h"

DPCTL_C_EXTERN_C_BEGIN

/*!
 * @brief Opaque pointer to a ``sycl::context``
 * @deprecated Use DpctlSyclContextRef
 */
typedef struct DPCTLOpaqueSyclContext *DPCTLSyclContextRef
    __attribute__((deprecated("Use DpctlSyclContextRef")));

/*!
 * @brief Opaque pointer to a ``sycl::device``
 * @deprecated Use DpctlSyclDeviceRef
 */
typedef struct DPCTLOpaqueSyclDevice *DPCTLSyclDeviceRef
    __attribute__((deprecated("Use DpctlSyclDeviceRef")));

/*!
 * @brief Opaque pointer to a ``sycl::device_selector``
 * @deprecated Use DpctlSyclDeviceSelectorRef
 */
typedef struct DPCTLOpaqueSyclDeviceSelector *DPCTLSyclDeviceSelectorRef
    __attribute__((deprecated("Use DpctlSyclDeviceSelectorRef")));

/*!
 * @brief Opaque pointer to a ``sycl::event``
 * @deprecated Use DpctlSyclEventRef
 */
typedef struct DPCTLOpaqueSyclEvent *DPCTLSyclEventRef
    __attribute__((deprecated("Use DpctlSyclEventRef")));
/*!
 * @brief Opaque pointer to a ``sycl::kernel``
 * @deprecated Use DpctlSyclKernelRef
 */
typedef struct DPCTLOpaqueSyclKernel *DPCTLSyclKernelRef
    __attribute__((deprecated("Use DpctlSyclKernelRef")));

/*!
 * @brief Opaque pointer to a ``sycl::platform``
 * @deprecated Use DpctlSyclPlatformRef
 */
typedef struct DPCTLOpaqueSyclPlatform *DPCTLSyclPlatformRef
    __attribute__((deprecated("Use DpctlSyclPlatformRef")));
/*!
 * @brief Opaque pointer to a ``sycl::program``
 * @deprecated Use DpctlSyclProgramRef
 */
typedef struct DPCTLOpaqueSyclProgram *DPCTLSyclProgramRef
    __attribute__((deprecated("Use DpctlSyclProgramRef")));

/*!
 * @brief Opaque pointer to a ``sycl::queue``
 * @deprecated Use DpctlSyclQueueRef
 */
typedef struct DPCTLOpaqueSyclQueue *DPCTLSyclQueueRef
    __attribute__((deprecated("Use DpctlSyclQueueRef")));

/*!
 * @brief Used to pass a ``sycl::usm`` memory opaquely through DPCTL interfaces.
 * @deprecated Use DpctlSyclUSMRef
 */
typedef struct DPCTLOpaqueSyclUSM *DPCTLSyclUSMRef
    __attribute__((deprecated("Use DpctlSyclUSMRef")));

//---------------------------- New API ---------------------------------------//

/*!
 * @brief Opaque pointer to a ``sycl::context``
 */
typedef struct DpctlOpaqueSyclContext *DpctlSyclContextRef;

/*!
 * @brief Opaque pointer to a ``sycl::device``
 */
typedef struct DpctlOpaqueSyclDevice *DpctlSyclDeviceRef;

/*!
 * @brief Opaque pointer to a ``sycl::device_selector``
 *
 */
typedef struct DpctlOpaqueSyclDeviceSelector *DpctlSyclDeviceSelectorRef;

/*!
 * @brief Opaque pointer to a ``sycl::event``
 *
 */
typedef struct DpctlOpaqueSyclEvent *DpctlSyclEventRef;

/*!
 * @brief Opaque pointer to a ``sycl::kernel``
 *
 */
typedef struct DpctlOpaqueSyclKernel *DpctlSyclKernelRef;

/*!
 * @brief Opaque pointer to a ``sycl::platform``
 *
 */
typedef struct DpctlOpaqueSyclPlatform *DpctlSyclPlatformRef;

/*!
 * @brief Opaque pointer to a ``sycl::program``
 *
 */
typedef struct DpctlOpaqueSyclProgram *DpctlSyclProgramRef;

/*!
 * @brief Opaque pointer to a ``sycl::queue``
 *
 */
typedef struct DpctlOpaqueSyclQueue *DpctlSyclQueueRef;

/*!
 * @brief Used to pass a ``sycl::usm`` memory opaquely through Dpctl interfaces.
 *
 */
typedef struct DpctlOpaqueSyclUSM *DpctlSyclUSMRef;

DPCTL_C_EXTERN_C_END
