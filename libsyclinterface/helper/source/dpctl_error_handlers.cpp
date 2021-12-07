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

#include "dpctl_error_handlers.h"
#include <cstring>

void DPCTL_AsyncErrorHandler::operator()(
    const cl::sycl::exception_list &exceptions)
{
    for (std::exception_ptr const &e : exceptions) {
        try {
            std::rethrow_exception(e);
        } catch (sycl::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
            auto err_code = e.get_cl_code();
            handler_(static_cast<int>(err_code));
        }
    }
}

namespace
{
int requested_verbosity_level(void)
{
    int requested_level = 0;

    const char *verbose = std::getenv("DPCTL_VERBOSITY");

    if (verbose) {
        if (!std::strncmp(verbose, "none", 4))
            requested_level = error_level::none;
        else if (!std::strncmp(verbose, "error", 5))
            requested_level = error_level::error;
        else if (!std::strncmp(verbose, "warning", 7))
            requested_level = error_level::warning;
    }

    return requested_level;
}
} // namespace

void error_handler(const std::exception &e,
                   const char *file_name,
                   const char *func_name,
                   int line_num,
                   error_level error_type)
{
    int requested_level = requested_verbosity_level();
    int error_level = static_cast<int>(error_type);

    if (requested_level >= error_level) {
        std::cerr << e.what() << " in " << func_name << " at " << file_name
                  << ":" << line_num << std::endl;
    }
}

void error_handler(const std::string &what,
                   const char *file_name,
                   const char *func_name,
                   int line_num,
                   error_level error_type)
{
    int requested_level = requested_verbosity_level();
    int error_level = static_cast<int>(error_type);

    if (requested_level >= error_level) {
        std::cerr << what << " In " << func_name << " at " << file_name << ":"
                  << line_num << std::endl;
    }
}
