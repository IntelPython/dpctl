//===- dpctl_sycl_ipc_memory_stubs.cpp - Stub IPC functions ---------------===//
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
/// Stub implementations of DPCTLIPCMem_* functions for builds where the
/// SYCL IPC memory extension is not available. These allow _memory.pyx to
/// always link; the functions return error codes at runtime.
///
//===----------------------------------------------------------------------===//

#include "dpctl_error_handlers.h"
#include "dpctl_sycl_ipc_memory_interface.h"
#include <cstddef>

int DPCTLIPCMem_GetHandle(__dpctl_keep DPCTLSyclUSMRef,
                          __dpctl_keep const DPCTLSyclContextRef,
                          char **,
                          size_t *)
{
    error_handler("IPC memory not supported in this build.", __FILE__, __func__,
                  __LINE__);
    return 1;
}

__dpctl_give DPCTLSyclUSMRef
DPCTLIPCMem_OpenHandle(const char *,
                       size_t,
                       __dpctl_keep const DPCTLSyclContextRef,
                       __dpctl_keep const DPCTLSyclDeviceRef)
{
    error_handler("IPC memory not supported in this build.", __FILE__, __func__,
                  __LINE__);
    return nullptr;
}

void DPCTLIPCMem_CloseHandle(__dpctl_keep DPCTLSyclUSMRef,
                             __dpctl_keep const DPCTLSyclContextRef)
{
    error_handler("IPC memory not supported in this build.", __FILE__, __func__,
                  __LINE__);
}

void DPCTLIPCMem_FreeHandleData(char *Data) { (void)Data; }
