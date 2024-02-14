//===--- dpctl_error_handler_types.h - Error handler callbacks   -*-C++-*- ===//
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
typedef void error_handler_callback(int err_code);
