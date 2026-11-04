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
/// This file implements the functions that defer releasing what offloaded work
/// uses until that work has completed: decrementing the reference counts of an
/// array of Python objects, or calling a release function another module
/// passes in.
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

namespace dpctl
{
namespace detail
{

/*!
 * @brief Name of the kernel submitted by `submit_keep_alive_marker`.
 */
class keep_alive_marker;

/*!
 * @brief An event that gates the release of objects used by work on `q`.
 *
 * Submits an empty kernel after `deps`. On an in-order queue the submission is
 * also ordered after everything already submitted there, so the event covers
 * uses of the objects that the caller never tracked.
 */
inline sycl::event
submit_keep_alive_marker(sycl::queue &q, const std::vector<sycl::event> &deps)
{
    return q.single_task<keep_alive_marker>(deps, []() {});
}

} // namespace detail
} // namespace dpctl

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
 * @brief The queue behind `QRef`, which must not be null.
 */
sycl::queue &unwrap_queue(DPCTLSyclQueueRef QRef)
{
    sycl::queue *q = dpctl::syclinterface::unwrap<sycl::queue>(QRef);
    if (!q) {
        throw std::invalid_argument("A queue is required");
    }
    return *q;
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
    using dpctl::syclinterface::wrap;

    try {
        sycl::queue &q = unwrap_queue(QRef);

        std::vector<sycl::event> depends = unwrap_events(depERefs, nDepERefs);
        std::vector<PyObject *> obj_vec(obj_array, obj_array + obj_array_size);

        static constexpr int result_ok = 0;

        const sycl::event marker =
            dpctl::detail::submit_keep_alive_marker(q, depends);

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

/*!
 * @brief Call `release(data)` once the work using what it releases is done.
 *
 * Submits an empty kernel to `QRef` after `depERefs`, as `async_dec_ref` does,
 * and assigns its event to the `sycl::event` behind `markerERef`. `release` is
 * called with the GIL held, on `dpctl`'s thread or on one that drains.
 *
 * @return 0 on success, and 1 if nothing was deferred, in which case `release`
 * is never called.
 */
inline int defer_release(DPCTLSyclQueueRef QRef,
                         DPCTLSyclEventRef *depERefs,
                         size_t nDepERefs,
                         void (*release)(void *),
                         void *data,
                         DPCTLSyclEventRef markerERef)
{
    using dpctl::syclinterface::unwrap;

    try {
        sycl::queue &q = unwrap_queue(QRef);
        sycl::event *marker = unwrap<sycl::event>(markerERef);
        if (!release || !marker) {
            throw std::invalid_argument("A release and an event are required");
        }

        *marker = dpctl::detail::submit_keep_alive_marker(
            q, unwrap_events(depERefs, nDepERefs));

        dpctl::detail::local_deferred_releases().defer(
            {*marker}, [release, data]() { release(data); });

        return 0;
    } catch (...) {
        // no exception may escape into the calling Cython code
        return 1;
    }
}
