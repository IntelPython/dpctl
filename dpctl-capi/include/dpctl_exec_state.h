//===- dpctl_exec_state.h - C API for service functions          -*-C++-*- ===//
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
/// The file declares a struct to store dpctl's error handler and other
/// execution state configurations.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"
#include "dpctl_error_handler_type.h"

DPCTL_C_EXTERN_C_BEGIN

/*!
 * @brief An opaque type to represent an "execution state" for the dpctl
 * sycl interface library.
 *
 * The execution state controls the behavior of functions exposed by
 * libDPCTLSyclInterface. For now, only error handling is controlled by an
 * execution state. Using a custom execution state, users can define how
 * exceptions in the C++ code are handled and propagated to callers. A default
 * execution state where all exceptions are caught and the error messages
 * printed to ``std::cerr`` is included for convenience.
 *
 */
typedef struct DpctlExecutionState *DpctlExecState;

/*!
 * @brief Create a new ``DpctlExecState`` object.
 *
 * @param    handler        An error handler function.
 * @return   A ``DpctlExecState`` opaque pointer.
 */
__dpctl_give DpctlExecState
dpctl_exec_state_create(error_handler_callback_fn handler);

/*!
 * @brief Create a default execution state that prints the error message to
 * ``std::cerr``.
 *
 * @return   A ``DpctlExecState`` opaque pointer.
 */
__dpctl_give DpctlExecState dpctl_exec_state_create_default();

/*!
 * @brief Delete an ``DpctlExecState`` opaque pointer.
 *
 * @param    DpctlExecState A ``DpctlExecState`` opaque pointer to be freed.
 */
void dpctl_exec_state_delete(__dpctl_take DpctlExecState state);

/*!
 * @brief Get the error handler defined in the ``DpctlExecState`` object.
 *
 * @param    state          An ``DpctlExecState`` object.
 * @return A error_handler_callback_fn function pointer that was stored inside
 * the DpctlExecState object.
 */
error_handler_callback_fn
dpctl_exec_state_get_error_handler(__dpctl_keep DpctlExecState state);

/*!
 * @brief Call the error handler defined in the ``DpctlExecState`` object.
 *
 * @param    state          An ``DpctlExecState`` object.
 * @param    err_code       An integer error code.
 * @param    err_msg        A C string corresponding to an error message.
 * @param    file_name      The file where the error occurred.
 * @param    func_name      The function name where the error occurred.
 * @param    line_num       The line number where the error occurred.
 */
void dpctl_exec_state_handle_error(__dpctl_keep DpctlExecState state,
                                   int err_code,
                                   __dpctl_keep const char *err_msg,
                                   __dpctl_keep const char *file_name,
                                   __dpctl_keep const char *func_name,
                                   int line_num);

DPCTL_C_EXTERN_C_END
