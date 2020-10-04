//===--- dppl_sycl_context_interface.h - DPPL-SYCL interface --*--C++ --*--===//
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
/// This header declares a C API to SYCL's sycl::context interface.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "dppl_data_types.h"
#include "dppl_sycl_types.h"
#include "dppl_sycl_platform_interface.h"
#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"
#include <stdbool.h>

DPPL_C_EXTERN_C_BEGIN

/*!
 * @brief Checks if two DPPLSyclContextRef objects point to the same
 * sycl::context.
 *
 * @param    CtxRef1       First opaque pointer to the sycl context.
 * @param    CtxRef2       Second opaque pointer to the sycl context.
 * @return   True if the underlying sycl::context are same, false otherwise.
 */
DPPL_API
bool DPPLContext_AreEq (__dppl_keep const DPPLSyclContextRef CtxRef1,
                        __dppl_keep const DPPLSyclContextRef CtxRef2);

/*!
 * @brief Returns true if this SYCL context is a host context.
 *
 * @param    CtxRef        An opaque pointer to a sycl::context.
 * @return   True if the SYCL context is a host context, else False.
 */
DPPL_API
bool DPPLContext_IsHost (__dppl_keep const DPPLSyclContextRef CtxRef);

/*!
 * @brief
 *
 * @param    CtxRef         An opaque pointer to a sycl::context.
 * @return   {return}       My Param doc
 */
DPPL_API
DPPLSyclBEType
DPPLContext_GetBackend (__dppl_keep const DPPLSyclContextRef CtxRef);

/*!
 * @brief Delete the pointer after casting it to sycl::context
 *
 * @param    CtxRef        The DPPLSyclContextRef pointer to be deleted.
 */
DPPL_API
void DPPLContext_Delete (__dppl_take DPPLSyclContextRef CtxRef);

DPPL_C_EXTERN_C_END
