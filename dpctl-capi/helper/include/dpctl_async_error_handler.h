//===-- dpctl_async_error_handler.h - An async error handler     -*-C++-*- ===//
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
/// A functor to use for passing an error handler callback function to sycl
/// context and queue contructors.
//===----------------------------------------------------------------------===//

#pragma once

#include "dpctl_error_handler_type.h"
#include <CL/sycl.hpp>

class DPCTL_AsycErrorHandler
{
    error_handler_callback *handler_ = nullptr;

public:
    DPCTL_AsycErrorHandler(error_handler_callback *err_handler)
        : handler_(err_handler)
    {
    }

    void operator()(const cl::sycl::exception_list &exceptions);
};