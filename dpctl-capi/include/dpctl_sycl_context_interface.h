//===-- dpctl_sycl_context_interface.h - C API for sycl::context -*-C++-*- ===//
//
//                      Data Parallel Control (dpCtl)
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
/// This header declares a C API to SYCL's sycl::context interface.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"
#include "dpctl_data_types.h"
#include "dpctl_sycl_enum_types.h"
#include "dpctl_sycl_types.h"
#include <stdbool.h>

DPCTL_C_EXTERN_C_BEGIN

/*!
 * @brief Checks if two DPCTLSyclContextRef objects point to the same
 * sycl::context.
 *
 * @param    CtxRef1       First opaque pointer to the sycl context.
 * @param    CtxRef2       Second opaque pointer to the sycl context.
 * @return   True if the underlying sycl::context are same, false otherwise.
 */
DPCTL_API
bool DPCTLContext_AreEq(__dpctl_keep const DPCTLSyclContextRef CtxRef1,
                        __dpctl_keep const DPCTLSyclContextRef CtxRef2);

/*!
 * @brief Returns true if this SYCL context is a host context.
 *
 * @param    CtxRef        An opaque pointer to a sycl::context.
 * @return   True if the SYCL context is a host context, else False.
 */
DPCTL_API
bool DPCTLContext_IsHost(__dpctl_keep const DPCTLSyclContextRef CtxRef);

/*!
 * @brief Returns the sycl backend for the DPCTLSyclContextRef pointer.
 *
 * @param    CtxRef         An opaque pointer to a sycl::context.
 * @return   The sycl backend for the DPCTLSyclContextRef returned as
 * a DPCTLSyclBackendType enum type.
 */
DPCTL_API
DPCTLSyclBackendType
DPCTLContext_GetBackend(__dpctl_keep const DPCTLSyclContextRef CtxRef);

/*!
 * @brief Delete the pointer after casting it to sycl::context
 *
 * @param    CtxRef        The DPCTLSyclContextRef pointer to be deleted.
 */
DPCTL_API
void DPCTLContext_Delete(__dpctl_take DPCTLSyclContextRef CtxRef);

DPCTL_C_EXTERN_C_END
