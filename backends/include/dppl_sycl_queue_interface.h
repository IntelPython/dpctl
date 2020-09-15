//===--- dppl_sycl_queue_interface.h - DPPL-SYCL interface ---*---C++ -*---===//
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
/// This header declares a C interface to sycl::queue member functions. Note
/// that sycl::queue constructors are not exposed in this interface. Instead,
/// users should use the functions in dppl_sycl_queue_manager.h.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "dppl_data_types.h"
#include "dppl_sycl_types.h"
#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"

DPPL_C_EXTERN_C_BEGIN

/*!
 * @brief Prints out information about the Sycl environment, such as
 * number of available platforms, number of activated queues, etc.
 */
DPPL_API
void DPPLDumpPlatformInfo ();

/*!
 * @brief Returns the Sycl context for the queue.
 *
 * @param    QRef           An opaque pointer to the sycl queue.
 * @return   A DPPLSyclContextRef pointer to the sycl context for the queue.
 */
DPPL_API
__dppl_give DPPLSyclContextRef
DPPLGetContextFromQueue (__dppl_keep const DPPLSyclQueueRef QRef);

/*!
 * @brief returns the Sycl device for the queue.
 *
 * @param    QRef           An opaque pointer to the sycl queue.
 * @return   A DPPLSyclDeviceRef pointer to the sycl device for the queue.
 */
DPPL_API
__dppl_give DPPLSyclDeviceRef
DPPLGetDeviceFromQueue (__dppl_keep const DPPLSyclQueueRef QRef);

/*!
 * @brief Delete the pointer after casting it to sycl::queue.
 *
 * @param    QRef           A DPPLSyclQueueRef pointer that gets deleted.
 */
DPPL_API
void DPPLDeleteSyclQueue (__dppl_take DPPLSyclQueueRef QRef);

DPPL_C_EXTERN_C_END
