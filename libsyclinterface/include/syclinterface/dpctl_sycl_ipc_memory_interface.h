//===- dpctl_sycl_ipc_memory_interface.h - C API for SYCL IPC mem -*-C++-*-===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2026 Intel Corporation
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
/// This header declares a C interface to
/// sycl::ext::oneapi::experimental::ipc::memory functions.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"
#include "dpctl_data_types.h"
#include "dpctl_sycl_types.h"

DPCTL_C_EXTERN_C_BEGIN

/**
 * @defgroup IPCMemoryInterface IPC Memory Interface
 */

/*!
 * @brief Get an IPC memory handle for a USM device pointer.
 *
 * Wraps ``sycl::ext::oneapi::experimental::ipc::memory::get()``.
 * The returned handle bytes are copied out.
 *
 * @param    Ptr         USM device pointer to export.
 * @param    CRef        Sycl context associated with the pointer.
 * @param    DataOut     [out] Pointer to receive a malloc'd byte buffer
 *                       containing the IPC handle data. Caller must free
 *                       with DPCTLIPCMem_FreeHandleData().
 * @param    SizeOut     [out] Pointer to receive the byte count of DataOut.
 * @return   0 on success, non-zero on failure.
 * @ingroup IPCMemoryInterface
 */
DPCTL_API
int DPCTLIPCMem_GetHandle(__dpctl_keep DPCTLSyclUSMRef Ptr,
                          __dpctl_keep const DPCTLSyclContextRef CRef,
                          char **DataOut,
                          size_t *SizeOut);

/*!
 * @brief Open an IPC memory handle in the receiving process.
 *
 * Wraps ``sycl::ext::oneapi::experimental::ipc::memory::open()``.
 *
 * @param    HandleData      Byte buffer from DPCTLIPCMem_GetHandle.
 * @param    HandleDataSize  Size of HandleData in bytes.
 * @param    CRef            Sycl context for the receiving side.
 * @param    DRef            Sycl device to map the memory on.
 * @return   A USM pointer to the IPC-mapped memory, or nullptr on failure.
 * @ingroup IPCMemoryInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclUSMRef
DPCTLIPCMem_OpenHandle(const char *HandleData,
                       size_t HandleDataSize,
                       __dpctl_keep const DPCTLSyclContextRef CRef,
                       __dpctl_keep const DPCTLSyclDeviceRef DRef);

/*!
 * @brief Close an IPC memory mapping opened by DPCTLIPCMem_OpenHandle.
 *
 * Wraps ``sycl::ext::oneapi::experimental::ipc::memory::close()``.
 *
 * @param    MappedPtr   The USM pointer returned by DPCTLIPCMem_OpenHandle.
 * @param    CRef        Sycl context used when opening the handle.
 * @ingroup IPCMemoryInterface
 */
DPCTL_API
void DPCTLIPCMem_CloseHandle(__dpctl_keep DPCTLSyclUSMRef MappedPtr,
                             __dpctl_keep const DPCTLSyclContextRef CRef);

/*!
 * @brief Free a handle data buffer returned by DPCTLIPCMem_GetHandle.
 *
 * @param    Data   Pointer previously returned via the DataOut parameter
 *                  of DPCTLIPCMem_GetHandle.
 * @ingroup IPCMemoryInterface
 */
DPCTL_API
void DPCTLIPCMem_FreeHandleData(char *Data);

DPCTL_C_EXTERN_C_END
