//===--- _async_dec_ref.hpp - Implements async DECREF ---------------------===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2026 Intel Corporation
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
/// N.B.: The reference counts are dropped by the thread `dpctl` keeps for
/// running deferred releases, or by a thread that drains them.
///
//===----------------------------------------------------------------------===//

#pragma once
#include <memory>
#include <stddef.h>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

#include "Python.h"

#include "_deferred_release_watcher.hpp"
#include "syclinterface/dpctl_data_types.h"
#include "syclinterface/dpctl_sycl_type_casters.hpp"

namespace
{

/*!
 * @brief Copy the events behind `nDepERefs` references into a vector.
 */
std::vector<sycl::event> unwrap_events(DPCTLSyclEventRef *depERefs,
                                       size_t nDepERefs)
{
    using dpctl::syclinterface::unwrap;

    std::vector<sycl::event> depends;
    depends.reserve(nDepERefs);
    for (size_t ev_id = 0; ev_id < nDepERefs; ++ev_id) {
        depends.push_back(*(unwrap<sycl::event>(depERefs[ev_id])));
    }

    return depends;
}

/*!
 * @brief Schedule DECREFs of `obj_vec` for once `depends` have completed.
 */
void submit_dec_ref(std::vector<PyObject *> obj_vec,
                    std::vector<sycl::event> depends)
{
    dpctl::detail::local_deferred_releases().defer(
        std::move(depends), [obj_vec = std::move(obj_vec)]() {
            for (PyObject *obj : obj_vec) {
                Py_DECREF(obj);
            }
        });
}

} // namespace

/*!
 * @brief Schedule DECREFs of `obj_array` for once `obj_array` is out of use.
 *
 * An empty kernel is submitted to `QRef` after `depERefs`, and the DECREFs wait
 * for it. On an in-order queue submission also orders it after the work already
 * submitted there, so the objects are covered regardless of providing dependent
 * events. `QRef` should be the queue running the work that uses the objects.
 * An event for the kernel is returned.
 *
 * Sets `*status` to 0 on success and 1 if scheduling threw, and returns nullptr
 * on failure. A failed call has scheduled nothing, so the caller still owns the
 * references it took.
 */
inline DPCTLSyclEventRef async_dec_ref(DPCTLSyclQueueRef QRef,
                                       PyObject **obj_array,
                                       size_t obj_array_size,
                                       DPCTLSyclEventRef *depERefs,
                                       size_t nDepERefs,
                                       int *status)
{
    using dpctl::syclinterface::unwrap;
    using dpctl::syclinterface::wrap;

    try {
        sycl::queue *q = unwrap<sycl::queue>(QRef);
        if (!q) {
            throw std::invalid_argument("A queue is required");
        }

        std::vector<sycl::event> depends = unwrap_events(depERefs, nDepERefs);
        std::vector<PyObject *> obj_vec(obj_array, obj_array + obj_array_size);

        static constexpr int result_ok = 0;

        const sycl::event marker =
            dpctl::detail::submit_keep_alive_marker(*q, depends);

        // allocated before scheduling, as failing afterwards could not be
        // reported: the caller would drop a reference the scheduled DECREFs own
        std::unique_ptr<sycl::event> e_ptr(new sycl::event(marker));

        submit_dec_ref(std::move(obj_vec), {marker});

        *status = result_ok;

        return wrap<sycl::event>(e_ptr.release());
    } catch (...) {
        // no exception may escape into the calling Cython code
        static constexpr int result_exception = 1;
        *status = result_exception;
        return nullptr;
    }
}
