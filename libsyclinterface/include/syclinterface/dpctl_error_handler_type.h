//===--- dpctl_error_handler_types.h - Error handler callbacks   -*-C++-*- ===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2021 Intel Corporation
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
 * @brief Function called for each SYCL asynchronous error.
 *
 * May be called on any thread, including concurrently.
 *
 * @param    err_code       Error code of the asynchronous error, or -1 if it
 *                          is not a ``sycl::exception``.
 * @param    err_msg        Message of the asynchronous error. Never NULL, and
 *                          valid only for the duration of the call.
 * @param    user_data      The ``user_data`` of the
 *                          ``DPCTLAsyncErrorHandlerSpec`` the queue or context
 *                          was created with.
 */
typedef void
error_handler_callback(int err_code, const char *err_msg, void *user_data);

/*!
 * @brief Asynchronous error handler to create a queue or a context with.
 *
 * The functions taking a ``DPCTLAsyncErrorHandlerSpec`` copy it, so it need
 * not outlive the call.
 */
typedef struct DPCTLAsyncErrorHandlerSpec
{
    /*! Called for each asynchronous error; may be NULL. */
    error_handler_callback *callback;
    /*! Passed to ``callback`` and ``release``. */
    void *user_data;
    /*!
     * If not NULL, called with ``user_data`` exactly once per call that was
     * given this handler, once no queue or context created by that call holds
     * the handler any longer, or as the call returns if it failed. May be
     * called on any thread.
     */
    void (*release)(void *user_data);
} DPCTLAsyncErrorHandlerSpec;
