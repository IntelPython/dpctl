//===-- dpctl_sycl_context_interface.h - C API for sycl::context -*-C++-*- ===//
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
/// This header declares a C API to SYCL's sycl::context interface.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"
#include "dpctl_data_types.h"
#include "dpctl_error_handler_type.h"
#include "dpctl_sycl_device_manager.h"
#include "dpctl_sycl_enum_types.h"
#include "dpctl_sycl_types.h"
#include <stdbool.h>

DPCTL_C_EXTERN_C_BEGIN

/**
 * @defgroup ContextInterface Context class C wrapper
 */

/*!
 * @brief Constructs a new SYCL context for the given SYCL device using the
 * optional async error handler and properties bit flags.
 *
 * @param    DRef           Opaque pointer to a SYCL device.
 * @param    handler        A callback function that will be invoked by the
 *                          async_handler used during context creation. Can be
 *                          NULL if no async_handler is needed.
 * @param    properties     An optional combination of bit flags to define
 *                          context properties. Currently, dpctl does not use
 *                          this argument.
 * @return   A new opaque pointer wrapping a SYCL context.
 * @ingroup ContextInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclContextRef
DPCTLContext_Create(__dpctl_keep const DPCTLSyclDeviceRef DRef,
                    error_handler_callback *handler,
                    int properties);

/*!
 * @brief Constructs a new SYCL context for the given vector of SYCL devices
 * using the optional async error handler and properties bit flags.
 *
 * @param    DVRef          An opaque pointer to a std::vector of
 *                          DPCTLSyclDeviceRef opaque pointers.
 * @param    handler        A callback function that will be invoked by the
 *                          async_handler used during context creation. Can be
 *                          NULL if no async_handler is needed.
 * @param    properties     An optional combination of bit flags to define
 *                          context properties. Currently, dpctl does not use
 *                          this argument.
 * @return   A new opaque pointer wrapping a SYCL context.
 * @ingroup ContextInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclContextRef
DPCTLContext_CreateFromDevices(__dpctl_keep const DPCTLDeviceVectorRef DVRef,
                               error_handler_callback *handler,
                               int properties);

/*!
 * @brief Checks if two DPCTLSyclContextRef objects point to the same
 * sycl::context.
 *
 * @param    CtxRef1       First opaque pointer to the sycl context.
 * @param    CtxRef2       Second opaque pointer to the sycl context.
 * @return   True if the underlying sycl::context are same, false otherwise.
 * @ingroup ContextInterface
 */
DPCTL_API
bool DPCTLContext_AreEq(__dpctl_keep const DPCTLSyclContextRef CtxRef1,
                        __dpctl_keep const DPCTLSyclContextRef CtxRef2);

/*!
 * @brief Returns a copy of the DPCTLSyclContextRef object.
 *
 * @param    CRef           DPCTLSyclContextRef object to be copied.
 * @return   A new DPCTLSyclContextRef created by copying the passed in
 * DPCTLSyclContextRef object.
 * @ingroup ContextInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclContextRef
DPCTLContext_Copy(__dpctl_keep const DPCTLSyclContextRef CRef);

/*!
 * @brief Returns the number of devices associated with sycl::context referenced
 * by DPCTLSyclContextRef object.
 *
 * @param    CRef           DPCTLSyclContexRef object to query.
 * @return   A positive count on success or zero on error.
 * @ingroup ContextInterface
 */
DPCTL_API
size_t DPCTLContext_DeviceCount(__dpctl_keep const DPCTLSyclContextRef CRef);

/*!
 * @brief Returns a vector of devices associated with sycl::context referenced
 * by DPCTLSyclContextRef object.
 *
 * @param    CRef           DPCTLSyclContexRef object to query.
 * @return   A DPCTLDeviceVectorRef with devices associated with given CRef.
 * @ingroup ContextInterface
 */
DPCTL_API
__dpctl_give DPCTLDeviceVectorRef
DPCTLContext_GetDevices(__dpctl_keep const DPCTLSyclContextRef CRef);

/*!
 * @brief Returns the sycl backend for the DPCTLSyclContextRef pointer.
 *
 * @param    CtxRef         An opaque pointer to a sycl::context.
 * @return   The sycl backend for the DPCTLSyclContextRef returned as
 * a DPCTLSyclBackendType enum type.
 * @ingroup ContextInterface
 */
DPCTL_API
DPCTLSyclBackendType
DPCTLContext_GetBackend(__dpctl_keep const DPCTLSyclContextRef CtxRef);

/*!
 * @brief Delete the pointer after casting it to sycl::context
 *
 * @param    CtxRef        The DPCTLSyclContextRef pointer to be deleted.
 * @ingroup ContextInterface
 */
DPCTL_API
void DPCTLContext_Delete(__dpctl_take DPCTLSyclContextRef CtxRef);

/*!
 * @brief Wrapper over std::hash<sycl::context>'s operator()
 *
 * @param    CtxRef        The DPCTLSyclContextRef pointer.
 * @return   Hash value of the underlying ``sycl::context`` instance.
 * @ingroup ContextInterface
 */
DPCTL_API
size_t DPCTLContext_Hash(__dpctl_keep DPCTLSyclContextRef CtxRef);

DPCTL_C_EXTERN_C_END
