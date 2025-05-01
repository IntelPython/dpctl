//===---- dpctl_sycl_extension_interface.h - C API for SYCL ext  -*-C++-*- ===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2025 Intel Corporation
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
/// This header declares a C interface to SYCL language extensions defined by
/// DPC++.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"
#include "dpctl_data_types.h"
#include "dpctl_error_handler_type.h"
#include "dpctl_sycl_enum_types.h"
#include "dpctl_sycl_types.h"

DPCTL_C_EXTERN_C_BEGIN

typedef struct RawWorkGroupMemoryTy
{
    size_t nbytes;
} RawWorkGroupMemory;

typedef struct DPCTLOpaqueSyclWorkGroupMemory *DPCTLSyclWorkGroupMemoryRef;

DPCTL_API
__dpctl_give DPCTLSyclWorkGroupMemoryRef
DPCTLWorkGroupMemory_Create(size_t nbytes);

DPCTL_API
void DPCTLWorkGroupMemory_Delete(__dpctl_take DPCTLSyclWorkGroupMemoryRef Ref);

DPCTL_API
bool DPCTLWorkGroupMemory_Available();

typedef struct DPCTLOpaqueSyclRawKernelArg *DPCTLSyclRawKernelArgRef;

DPCTL_API
__dpctl_give DPCTLSyclRawKernelArgRef DPCTLRawKernelArg_Create(void *bytes,
                                                               size_t count);

DPCTL_API
void DPCTLRawKernelArg_Delete(__dpctl_take DPCTLSyclRawKernelArgRef Ref);

DPCTL_API
bool DPCTLRawKernelArg_Available();

DPCTL_C_EXTERN_C_END
