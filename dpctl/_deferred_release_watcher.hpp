//===--- _deferred_release_watcher.hpp - Runs deferred releases -----------===//
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
/// This file implements the list of deferred releases that `dpctl` owns and the
/// thread that runs them.
///
//===----------------------------------------------------------------------===//

#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "Python.h"

#include "detail/deferred_releases.hpp"

namespace dpctl
{
namespace detail
{

/*!
 * @brief The deferred releases submitted.
 *
 * Reached from elsewhere through the module's `DeferredReleases_Get`.
 */
inline DeferredReleases &local_deferred_releases()
{
    // leaked: releasing what the list holds runs Python code, which a static
    // destructor could not do once the interpreter has shut down
    static DeferredReleases *instance = new DeferredReleases();
    return *instance;
}

/*!
 * @brief Whether the interpreter can still be called into.
 */
inline bool interpreter_is_live()
{
    const bool initialized = Py_IsInitialized();
#if PY_VERSION_HEX < 0x30d0000
    return initialized && !_Py_IsFinalizing();
#else
    return initialized && !Py_IsFinalizing();
#endif
}

/*!
 * @brief Runs the deferred releases, on a thread of its own.
 *
 * Looks over the list for releases that have come due and takes the GIL to run
 * them.
 *
 * The thread is ended by the `atexit` hook `dpctl` registers at import, or by
 * the destructor in a process that never finalizes the interpreter. In either
 * case, it stops running releases once shutdown has begun, leaving whatever is
 * still held for the process to reclaim.
 */
class DeferredReleaseWatcher
{
public:
    /*!
     * @brief Start the thread, if it is not already running.
     */
    void start()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (thread_.joinable()) {
            return;
        }

        local_deferred_releases().resume_waiting();
        finished().store(false, std::memory_order_release);

        try {
            thread_ = std::thread([]() { watch(); });
        } catch (...) {
            // without the thread the releases are left to the drains, so
            // failing to start one is not fatal
        }
    }

    /*!
     * @brief End the thread and wait for it, if it is running.
     *
     * Expects the caller not to hold the GIL, which the thread may be waiting
     * for..
     */
    void stop()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!thread_.joinable()) {
            return;
        }

        local_deferred_releases().stop_waiting();
        thread_.join();
    }

    /*!
     * @brief End the thread as the module's static objects are destroyed.
     *
     * Only ends the thread in a process that never finalizes the interpreter,
     * as the `atexit` hook ends it while the interpreter is still up.
     *
     * The thread is waited for rather than joined on Windows, where a thread
     * cannot exit without the loader lock that unloading a module holds.
     */
    ~DeferredReleaseWatcher()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        end_without_hanging(/* may_join */ joining_at_unload_is_safe);
    }

    DeferredReleaseWatcher(const DeferredReleaseWatcher &) = delete;
    DeferredReleaseWatcher &operator=(const DeferredReleaseWatcher &) = delete;

private:
    friend DeferredReleaseWatcher &deferred_release_watcher();

    DeferredReleaseWatcher() = default;

    using clock = DeferredReleases::clock;

    /*!
     * @brief How long the thread sleeps between passes, at the least.
     */
    static constexpr clock::duration min_poll_interval =
        std::chrono::milliseconds(5);

    /*!
     * @brief How many times a pass over the list its own duration buys.
     *
     * The sleep is stretched with the cost of a pass, which grows with the
     * number of items, so the thread costs about a `1 / poll_share` of a core.
     */
    static constexpr clock::duration::rep poll_share = 20;

    /*!
     * @brief How long an exit waits for the thread before detaching it.
     */
    static constexpr clock::duration abandon_after =
        std::chrono::milliseconds(200);

    /*!
     * @brief Whether a thread may be joined as this module is unloaded.
     *
     * False on Windows, where a thread cannot exit without the loader lock that
     * unloading holds.
     */
#ifdef _WIN32
    static constexpr bool joining_at_unload_is_safe = false;
#else
    static constexpr bool joining_at_unload_is_safe = true;
#endif

    /*!
     * @brief Stop the thread and let go of it, waiting only so long for it.
     *
     * A thread that does not report itself finished within `abandon_after` is
     * detached, so that an exit cannot hang on one the interpreter parked for
     * asking for the GIL. Expects `mutex_` to be held, and the caller not to
     * hold the GIL.
     */
    void end_without_hanging(bool may_join)
    {
        if (!thread_.joinable()) {
            return;
        }

        local_deferred_releases().stop_waiting();

        // waited for whether or not it may be joined: the report says the
        // thread is done with this object, which is what makes detaching safe
        const bool reported_finished = wait_for_thread(abandon_after);

        if (may_join && reported_finished) {
            thread_.join();
        }
        else {
            thread_.detach();
        }
    }

    /*!
     * @brief Whether the thread has left its loop.
     *
     * In static storage, so that a detached thread can report itself without
     * reaching a destroyed object.
     */
    static std::atomic<bool> &finished()
    {
        static std::atomic<bool> flag{false};
        return flag;
    }

    /*!
     * @brief Run the releases that come due, until the list is stopped or the
     * interpreter begins shutting down.
     */
    static void watch()
    {
        DeferredReleases &releases = local_deferred_releases();
        clock::duration poll_interval = min_poll_interval;
        std::vector<std::function<void()>> due;

        while (releases.wait_for_due(poll_interval)) {
            const clock::time_point started = clock::now();
            const bool any = releases.collect_due(due);
            const clock::duration looked_for = clock::now() - started;

            poll_interval =
                std::max(min_poll_interval, looked_for * poll_share);

            if (!any) {
                continue;
            }

            // what was collected is not run, like everything else still held
            if (!interpreter_is_live()) {
                break;
            }

            const PyGILState_STATE gstate = PyGILState_Ensure();
            // shutdown may have begun while the GIL was waited for
            const bool live = interpreter_is_live();
            if (live) {
                releases.run_collected(due);
            }
            PyGILState_Release(gstate);

            if (!live) {
                break;
            }
        }

        finished().store(true, std::memory_order_release);
    }

    /*!
     * @brief Whether the thread left its loop within `timeout`.
     */
    static bool wait_for_thread(clock::duration timeout)
    {
        const clock::time_point deadline = clock::now() + timeout;

        while (!finished().load(std::memory_order_acquire)) {
            if (clock::now() >= deadline) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        return true;
    }

    std::thread thread_;
    std::mutex mutex_;
};

/*!
 * @brief The thread running the deferred releases.
 *
 * Destroyed with the module's other static objects, which is what ends the
 * thread when a process exits without having stopped it.
 */
inline DeferredReleaseWatcher &deferred_release_watcher()
{
    static DeferredReleaseWatcher instance;
    return instance;
}

} // namespace detail
} // namespace dpctl

/*!
 * @brief Address of the module's `DeferredReleases`.
 *
 * Returns nullptr if the list could not be created.
 */
inline void *deferred_releases_ptr()
{
    try {
        return static_cast<void *>(&dpctl::detail::local_deferred_releases());
    } catch (...) {
        // no exception may escape into Cython
        return nullptr;
    }
}

/*!
 * @brief Run the releases that have come due, dropping the references held.
 *
 * Expects the caller to hold the GIL, as a release runs arbitrary Python code.
 *
 * @return Whether there were any.
 */
inline bool drain_deferred_releases()
{
    try {
        return dpctl::detail::local_deferred_releases().drain();
    } catch (...) {
        // no exception may escape into the calling Cython code
        return false;
    }
}

/*!
 * @brief Start running deferred releases on a thread of `dpctl`'s.
 */
inline void start_deferred_release_watcher()
{
    try {
        dpctl::detail::deferred_release_watcher().start();
    } catch (...) {
        // no exception may escape into the calling Cython code
    }
}

/*!
 * @brief Stop running deferred releases, and wait for the thread to end.
 *
 * Expects the caller not to hold the GIL.
 */
inline void stop_deferred_release_watcher()
{
    try {
        dpctl::detail::deferred_release_watcher().stop();
    } catch (...) {
        // no exception may escape into the calling Cython code
    }
}
