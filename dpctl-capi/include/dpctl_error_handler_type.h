//===--- dpctl_error_handler_types.h - Error handler callbacks   -*-C++-*- ===//
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
/// Defines types for callback functions to use for error handling in dpctl.
///
//===----------------------------------------------------------------------===//

#pragma once

/*!
 * @brief    Type of function to be used in SYCL async error handler provide by
 * DPCTL.
 *
 * @param    err_code       Error code extracted from an SYCL asynchronous
 * error.
 */
typedef void error_handler_callback(int err_code) __attribute__((
    deprecated("the function does not allow passing in an error string, use "
               "'error_handler_callback_fn' instead!!!")));

/*!
 * @brief Type signature required for an error handler callback function.
 *
 * @param    err_code       An integer error code.
 * @param    err_msg        A C string corresponding to an error message.
 * @param    file_name      The file where the error occurred.
 * @param    func_name      The function name where the error occurred.
 * @param    line_num       The line number where the error occurred.
 */
typedef void (*error_handler_callback_fn)(int err_code,
                                          const char *err_msg,
                                          const char *file_name,
                                          const char *func_name,
                                          int line_num);
