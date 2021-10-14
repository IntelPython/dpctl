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
#include <iomanip>
#include <iostream>

void DPCTL_AsyncErrorHandler::operator()(
    const cl::sycl::exception_list &exceptions)
{
    for (std::exception_ptr const &e : exceptions) {
        try {
            std::rethrow_exception(e);
        } catch (cl::sycl::exception const &e) {
            std::cerr << "Caught asynchronous SYCL exception:\n"
                      << e.what() << std::endl;
            // FIXME: Change get_cl_code() to code() once DPCPP supports it.
            auto err_code = e.get_cl_code();
            handler_(err_code);
        }
    }
}

void DefaultErrorHandler::handler(int err_code,
                                  const char *err_msg,
                                  const char *file_name,
                                  const char *func_name,
                                  int line_num)
{
    std::stringstream ss;

    ss << "Dpctl-Error ";
    if (file_name)
        ss << "on " << file_name << " ";

    if (func_name)
        ss << "at " << func_name << " ";

    if (line_num)
        ss << "on line " << line_num << ".";

    ss << " (" << err_code << ")";

    if (err_msg)
        ss << " " << err_msg << '\n';

    std::cerr << ss.str();
}
