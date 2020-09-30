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
 * @brief Returns the Sycl context for the queue.
 *
 * @param    QRef           An opaque pointer to the sycl queue.
 * @return   A DPPLSyclContextRef pointer to the sycl context for the queue.
 */
DPPL_API
__dppl_give DPPLSyclContextRef
DPPLQueue_GetContext (__dppl_keep const DPPLSyclQueueRef QRef);

/*!
 * @brief returns the Sycl device for the queue.
 *
 * @param    QRef           An opaque pointer to the sycl queue.
 * @return   A DPPLSyclDeviceRef pointer to the sycl device for the queue.
 */
DPPL_API
__dppl_give DPPLSyclDeviceRef
DPPLQueue_GetDevice (__dppl_keep const DPPLSyclQueueRef QRef);

/*!
 * @brief Delete the pointer after casting it to sycl::queue.
 *
 * @param    QRef           A DPPLSyclQueueRef pointer that gets deleted.
 */
DPPL_API
void DPPLQueue_Delete (__dppl_take DPPLSyclQueueRef QRef);

/*!
 * @brief C-API wrapper for sycl::queue::memcpy. It waits an event.
 *
 * @param    QRef           An opaque pointer to the sycl queue.
 * @param    Dest           An USM pointer to the destination memory.
 * @param    Src            An USM pointer to the source memory.
 * @param    Count          A number of bytes to copy.
 */
DPPL_API
void DPPLQueue_Memcpy (__dppl_keep const DPPLSyclQueueRef QRef,
                       void *Dest, const void *Src, size_t Count);

DPPL_C_EXTERN_C_END
