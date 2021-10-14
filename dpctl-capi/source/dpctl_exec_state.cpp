//===- dpctl_exec_state.cpp - Implements C API for sycl::context         ---==//
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
/// This file implements the data types and functions declared in
/// dpctl_exec_state.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_exec_state.h"
#include "../helper/include/dpctl_error_handlers.h"
#include "Support/CBindingWrapping.h"
#include <iomanip>
#include <iostream>

namespace
{

/*!
 * @brief The execution state that is passed to every libDPCTLSyclInterface
 * function.
 *
 * The ``ExecutionState`` class is a concrete implementation of the
 * `DpctlExecState`` opaque type.
 *
 */
class ExecutionState
{
    error_handler_callback_fn handler_;

public:
    /*!
     * @brief Construct a new ``ExecutionState`` object using the default error
     * handler object.
     *
     */
    ExecutionState() : handler_(DefaultErrorHandler::handler){};
    /*!
     * @brief Construct a new ``ExecutionState`` object with the provided error
     * handler function.
     *
     * @param    handler        Error handler function to be used by the
     * instance of ``ExecutionState``.
     */
    explicit ExecutionState(error_handler_callback_fn handler)
        : handler_(handler)
    {
    }

    void operator()(int err_code,
                    const char *err_msg,
                    const char *file_name,
                    const char *func_name,
                    int line_num) const
    {
        handler_(err_code, err_msg, file_name, func_name, line_num);
    }

    error_handler_callback_fn get_handler() const
    {
        return handler_;
    }
};

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(ExecutionState, DpctlExecState);

} // namespace

__dpctl_give DpctlExecState
dpctl_exec_state_create(error_handler_callback_fn handler)
{
    try {
        ExecutionState *state = new ExecutionState(handler);
        return wrap(state);
    } catch (std::bad_alloc const &ba) {
        std::cerr << ba.what() << '\n';
        std::terminate();
    }
}

__dpctl_give DpctlExecState dpctl_exec_state_create_default()
{
    try {
        ExecutionState *state = new ExecutionState();
        return wrap(state);
    } catch (std::bad_alloc const &ba) {
        std::cerr << ba.what() << '\n';
        std::terminate();
    }
}

void dpctl_exec_state_delete(__dpctl_take DpctlExecState state)
{
    delete unwrap(state);
}

error_handler_callback_fn
dpctl_exec_state_get_error_handler(__dpctl_keep DpctlExecState state)
{
    auto ES = unwrap(state);
    if (!ES) {
        std::cerr << "Execution state is corrupted. Abort!\n";
        std::terminate();
    }

    return ES->get_handler();
}

void dpctl_exec_state_handle_error(__dpctl_keep DpctlExecState state,
                                   int err_code,
                                   __dpctl_keep const char *err_msg,
                                   __dpctl_keep const char *file_name,
                                   __dpctl_keep const char *func_name,
                                   int line_num)
{
    auto ES = unwrap(state);
    if (!ES) {
        std::cerr << "Execution state is corrupted. Abort!\n";
        std::terminate();
    }

    (*ES)(err_code, err_msg, file_name, func_name, line_num);
}
