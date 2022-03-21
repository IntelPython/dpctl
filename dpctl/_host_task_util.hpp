//===--- _host_tasl_util.hpp - Implements async DECREF =//
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
/// This file implements a utility function to schedule host task to a sycl
/// queue depending on given array of sycl events to decrement reference counts
/// for the given array of Python objects.
///
/// N.B.: The host task attempts to acquire GIL, so queue wait, event wait and
/// other synchronization mechanisms should be called after releasing the GIL to
/// avoid deadlocks.
///
//===----------------------------------------------------------------------===//

#include "Python.h"
#include "syclinterface/dpctl_data_types.h"
#include <CL/sycl.hpp>

int async_dec_ref(DPCTLSyclQueueRef QRef,
                  PyObject **obj_array,
                  size_t obj_array_size,
                  DPCTLSyclEventRef *ERefs,
                  size_t nERefs)
{

    sycl::queue *q = reinterpret_cast<sycl::queue *>(QRef);

    std::vector<PyObject *> obj_vec;
    obj_vec.reserve(obj_array_size);
    for (size_t obj_id = 0; obj_id < obj_array_size; ++obj_id) {
        obj_vec.push_back(obj_array[obj_id]);
    }

    try {
        q->submit([&](sycl::handler &cgh) {
            for (size_t ev_id = 0; ev_id < nERefs; ++ev_id) {
                cgh.depends_on(
                    *(reinterpret_cast<sycl::event *>(ERefs[ev_id])));
            }
            cgh.host_task([obj_array_size, obj_vec]() {
                // if the main thread has not finilized the interpreter yet
                if (Py_IsInitialized() && !_Py_IsFinalizing()) {
                    PyGILState_STATE gstate;
                    gstate = PyGILState_Ensure();
                    for (size_t i = 0; i < obj_array_size; ++i) {
                        Py_DECREF(obj_vec[i]);
                    }
                    PyGILState_Release(gstate);
                }
            });
        });
    } catch (const std::exception &e) {
        return 1;
    }

    return 0;
}
