//===--- _async_dec_ref.hpp - Implements async DECREF ---------------------===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2022 Intel Corporation
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
/// This file implements a utility function to decrement reference counts for a
/// given array of Python objects once a given array of sycl events has
/// completed.
///
/// N.B.: The deferred work acquires the GIL, so queue wait, event wait and
/// other synchronization mechanisms should be called after releasing the GIL to
/// avoid deadlocks.
///
//===----------------------------------------------------------------------===//

#pragma once
#include <exception>
#include <stddef.h>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

#include "Python.h"

#include "detail/keep_alive_pool.hpp"
#include "syclinterface/dpctl_data_types.h"
#include "syclinterface/dpctl_sycl_type_casters.hpp"

DPCTLSyclEventRef async_dec_ref(DPCTLSyclQueueRef QRef,
                                PyObject **obj_array,
                                size_t obj_array_size,
                                DPCTLSyclEventRef *depERefs,
                                size_t nDepERefs,
                                int *status)
{
    using dpctl::syclinterface::unwrap;
    using dpctl::syclinterface::wrap;

    // `QRef` is kept in the signature for API compatibility
    (void)QRef;

    std::vector<PyObject *> obj_vec(obj_array, obj_array + obj_array_size);

    try {
        std::vector<sycl::event> depends;
        depends.reserve(nDepERefs);
        for (size_t ev_id = 0; ev_id < nDepERefs; ++ev_id) {
            depends.push_back(*(unwrap<sycl::event>(depERefs[ev_id])));
        }

        dpctl::detail::KeepAlivePool::get().submit(
            std::move(depends),
            [obj_array_size, obj_vec = std::move(obj_vec)]() {
                const bool initialized = Py_IsInitialized();
#if PY_VERSION_HEX < 0x30d0000
                const bool finalizing = _Py_IsFinalizing();
#else
                const bool finalizing = Py_IsFinalizing();
#endif
                // if the main thread has not finalized the interpreter yet
                if (initialized && !finalizing) {
                    PyGILState_STATE gstate;
                    gstate = PyGILState_Ensure();
                    for (size_t i = 0; i < obj_array_size; ++i) {
                        Py_DECREF(obj_vec[i]);
                    }
                    PyGILState_Release(gstate);
                }
            });

        static constexpr int result_ok = 0;

        *status = result_ok;
        // return a dummy event for API compatibility
        auto e_ptr = new sycl::event();
        return wrap<sycl::event>(e_ptr);
    } catch (const std::exception &e) {
        static constexpr int result_std_exception = 1;

        *status = result_std_exception;
        return nullptr;
    }

    static constexpr int result_other_abnormal = 2;

    *status = result_other_abnormal;
    return nullptr;
}
