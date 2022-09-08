//===-- dpctl_async_error_handler.h - An async error handler     -*-C++-*- ===//
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
/// A functor to use for passing an error handler callback function to sycl
/// context and queue contructors.
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/DllExport.h"
#include "dpctl_error_handler_type.h"
#include <CL/sycl.hpp>

/*!
 * @brief Functor class used by DPCTL to handle SYCL asynchronous errors.
 */
class DPCTL_API DPCTL_AsyncErrorHandler
{
    error_handler_callback *handler_ = nullptr;

public:
    DPCTL_AsyncErrorHandler(error_handler_callback *err_handler)
        : handler_(err_handler)
    {
    }

    void operator()(const sycl::exception_list &exceptions);
};

enum error_level : int
{
    none = 0,
    error = 1,
    warning = 2
};

void error_handler(const std::exception &e,
                   const char *file_name,
                   const char *func_name,
                   int line_num,
                   error_level error_type = error_level::error);

void error_handler(const std::string &what,
                   const char *file_name,
                   const char *func_name,
                   int line_num,
                   error_level error_type = error_level::warning);
