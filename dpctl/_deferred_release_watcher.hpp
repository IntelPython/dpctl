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

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

#include "Python.h"

#include "_deferred_releases.hpp"

namespace dpctl
{
namespace detail
{

inline void start_watcher_if_enabled();

/*!
 * @brief The deferred releases submitted.
 *
 * Other modules reach it through `SyclQueue_DeferRelease`. The first release
 * deferred starts `dpctl`'s thread.
 */
inline DeferredReleases &local_deferred_releases()
{
    // leaked: releasing what the list holds runs Python code, which a static
    // destructor could not do once the interpreter has shut down
    static DeferredReleases *instance =
        new DeferredReleases(&start_watcher_if_enabled);
    return *instance;
}

/*!
 * @brief Runs the deferred releases, on a thread of its own.
 *
 * Looks over the list for releases that have come due and takes the GIL to run
 * them.
 *
 * The thread is started by the first release deferred, and ended by the
 * `atexit` hook `dpctl` registers at import, or by the destructor in a process
 * that never finalizes the interpreter. In either case, it stops running
 * releases once shutdown has begun, leaving whatever is still held for the
 * process to reclaim.
 */
class DeferredReleaseWatcher
{
public:
    /*!
     * @brief Start the thread, and start it again whenever it is needed.
     *
     * Waits up to `stop_timeout` for a thread that a timed-out `stop` let go
     * of, so expects the caller not to hold the GIL.
     */
    void start()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);

            enabled_ = true;
            if (thread_.joinable() || stopping_ ||
                finished().load(std::memory_order_acquire))
            {
                start_nolock();
                return;
            }
        }

        wait_for_thread(stop_timeout);

        std::lock_guard<std::mutex> lock(mutex_);
        if (enabled_) {
            start_nolock();
        }
    }

    /*!
     * @brief Start the thread if it is not running and has not been stopped.
     */
    void start_if_enabled()
    {
        if (!finished().load(std::memory_order_acquire)) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        if (enabled_) {
            start_nolock();
        }
    }

    /*!
     * @brief End the thread, and keep it from being started until `start`.
     *
     * Waits up to `stop_timeout` for the release in progress, if any, and
     * detaches the thread if it has not finished by then. The releases not
     * yet run are left on the list.
     *
     * Expects the caller not to hold the GIL, which the thread may be waiting
     * for.
     */
    void stop()
    {
        std::thread thread;
        {
            std::lock_guard<std::mutex> lock(mutex_);

            enabled_ = false;
            if (!thread_.joinable()) {
                return;
            }
            stopping_ = true;
            thread = std::move(thread_);
        }

        // not under `mutex_`, which a thread holding the GIL may be waiting on
        end_without_hanging(thread, /* may_join */ true, stop_timeout);

        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = false;
        // a `start` made while stopping
        if (enabled_) {
            start_nolock();
        }
    }

    /*!
     * @brief Make the thread startable in the child of a `fork`.
     *
     * The thread is not copied into the child, so its handle is dropped
     * without being joined or detached, and the lock it may have held is
     * replaced. The list is reset as well, and the thread is started by the
     * next release deferred.
     */
    void reset_after_fork()
    {
        new (&thread_) std::thread();
        new (&mutex_) std::mutex();
        stopping_ = false;
        finished().store(true, std::memory_order_release);

        local_deferred_releases().reset_after_fork();
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

        enabled_ = false;
        unloading().store(true, std::memory_order_release);
        end_without_hanging(thread_, joining_at_unload_is_safe, abandon_after);
    }

    DeferredReleaseWatcher(const DeferredReleaseWatcher &) = delete;
    DeferredReleaseWatcher &operator=(const DeferredReleaseWatcher &) = delete;

private:
    friend DeferredReleaseWatcher &deferred_release_watcher();

    DeferredReleaseWatcher() = default;

    using clock = DeferredReleases::clock;

    /*!
     * @brief Whether the interpreter can still be called into.
     */
    static bool interpreter_is_live()
    {
        const bool initialized = Py_IsInitialized();
#if PY_VERSION_HEX < 0x30d0000
        return initialized && !_Py_IsFinalizing();
#else
        return initialized && !Py_IsFinalizing();
#endif
    }

    /*!
     * @brief How long the thread sleeps between passes, at the least.
     */
    static constexpr clock::duration min_poll_interval =
        std::chrono::milliseconds(5);

    /*!
     * @brief How many times a pass over the list its own duration buys.
     *
     * The sleep is stretched with the cost of a pass, which grows with the
     * number of items, so the thread costs about a `1 / poll_share` of a core
     * however many there are.
     */
    static constexpr clock::duration::rep poll_share = 20;

    /*!
     * @brief How long `stop` waits for the release in progress.
     */
    static constexpr clock::duration stop_timeout = std::chrono::seconds(1);

    /*!
     * @brief How long an exit waits for the thread before detaching it.
     */
    static constexpr clock::duration abandon_after =
        std::chrono::milliseconds(200);

    /*!
     * @brief The thread's name in Python's `threading`.
     */
    static constexpr const char *thread_name = "dpctl-deferred-releases";

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
     * @brief Start the thread, unless one is running or still finishing.
     *
     * A thread that ended on its own is joined and replaced, unless the
     * interpreter is shutting down. Expects `mutex_` to be held.
     */
    void start_nolock()
    {
        if (stopping_ || !finished().load(std::memory_order_acquire)) {
            return;
        }

        if (thread_.joinable()) {
            // a new thread would end at once
            if (!interpreter_is_live()) {
                return;
            }
            thread_.join();
        }

        local_deferred_releases().resume_waiting();
        finished().store(false, std::memory_order_release);

        try {
            thread_ = std::thread([]() { watch(); });
        } catch (...) {
            // without the thread the releases are left to the drains, so
            // failing to start one is not fatal
            finished().store(true, std::memory_order_release);
        }
    }

    /*!
     * @brief Stop `thread` and let go of it, waiting only `timeout` for it.
     *
     * A thread that does not report itself finished in time is detached, so
     * that an exit cannot hang on a release that blocks, or on a thread the
     * interpreter parked for asking for the GIL. Expects the caller not to
     * hold the GIL.
     */
    static void end_without_hanging(std::thread &thread,
                                    bool may_join,
                                    clock::duration timeout)
    {
        if (!thread.joinable()) {
            return;
        }

        local_deferred_releases().stop_waiting();

        // waited for whether or not it may be joined: the report says the
        // thread is done with this object, which is what makes detaching safe
        const bool done = has_exited(thread) || wait_for_thread(timeout);

        if (may_join && done) {
            thread.join();
        }
        else {
            thread.detach();
        }
    }

    /*!
     * @brief Whether no thread is running the loop.
     *
     * In static storage, so that a detached thread can report itself without
     * reaching a destroyed object.
     */
    static std::atomic<bool> &finished()
    {
        static std::atomic<bool> flag{true};
        return flag;
    }

    /*!
     * @brief Whether the module's static objects are being destroyed.
     *
     * The interpreter may then be up but not running, so the thread must not
     * wait for the GIL.
     */
    static std::atomic<bool> &unloading()
    {
        static std::atomic<bool> flag{false};
        return flag;
    }

    /*!
     * @brief Name the calling thread in Python's `threading`.
     *
     * Expects the caller to hold the GIL. Failing to is not reported.
     */
    static void name_python_thread()
    {
        PyObject *threading = PyImport_ImportModule("threading");
        PyObject *current =
            threading
                ? PyObject_CallMethod(threading, "current_thread", nullptr)
                : nullptr;
        if (current) {
            PyObject *name = PyUnicode_FromString(thread_name);
            if (name) {
                static_cast<void>(
                    PyObject_SetAttrString(current, "name", name));
                Py_DECREF(name);
            }
        }
        Py_XDECREF(current);
        Py_XDECREF(threading);
        PyErr_Clear();
    }

    /*!
     * @brief Run the releases that come due, until the list is stopped, the
     * interpreter begins shutting down, or the list throws.
     *
     * A thread ended by an exception is replaced by the next release deferred.
     */
    static void watch()
    {
        std::unique_ptr<std::vector<std::function<void()>>> due;

        try {
            due = std::make_unique<std::vector<std::function<void()>>>();
            run_due_until_stopped(*due);
        }
#if defined(__GLIBCXX__)
        catch (abi::__forced_unwind &)
        {
            // the interpreter ending this thread, which must not be stopped
            static_cast<void>(due.release());
            finished().store(true, std::memory_order_release);
            throw;
        }
#endif
        catch (...)
        {
            // an exception escaping the thread would terminate the process.
            // What was collected is leaked rather than destroyed, as the
            // destructors of releases not run may need the GIL
            static_cast<void>(due.release());
        }

        finished().store(true, std::memory_order_release);
    }

    /*!
     * @brief The loop run by `watch`, collecting releases onto `due`.
     *
     * The Python thread state made for the first releases run is kept until
     * the loop ends, so that the thread is one Python thread throughout.
     */
    static void run_due_until_stopped(std::vector<std::function<void()>> &due)
    {
        DeferredReleases &releases = local_deferred_releases();
        clock::duration poll_interval = min_poll_interval;

        bool keeps_thread_state = false;
        PyGILState_STATE kept = PyGILState_UNLOCKED;

        while (releases.wait_for_due(poll_interval)) {
            const clock::time_point started = clock::now();
            const bool any = releases.collect_due(due);
            const clock::duration looked_for = clock::now() - started;

            poll_interval = looked_for * poll_share;
            if (poll_interval < min_poll_interval) {
                poll_interval = min_poll_interval;
            }

            if (!any) {
                continue;
            }

            // what was collected is not run, like everything else still held,
            // and the thread state is left to the interpreter's shutdown
            if (!interpreter_is_live()) {
                releases.put_back(due);
                return;
            }

            const PyGILState_STATE gstate = PyGILState_Ensure();
            // shutdown may have begun while the GIL was waited for
            const bool live = interpreter_is_live();
            try {
                if (live) {
                    if (!keeps_thread_state) {
                        kept = PyGILState_Ensure();
                        keeps_thread_state = true;
                        name_python_thread();
                    }
                    releases.run_collected(due, /* until_stopped */ true);
                }
            }
#if defined(__GLIBCXX__)
            catch (abi::__forced_unwind &)
            {
                // the GIL was dropped by the interpreter ending this thread
                throw;
            }
#endif
            catch (...)
            {
                // `kept` first, so that the release deleting the thread state
                // is the one dropping the GIL
                if (keeps_thread_state) {
                    PyGILState_Release(kept);
                }
                PyGILState_Release(gstate);
                throw;
            }
            PyGILState_Release(gstate);

            if (!live) {
                releases.put_back(due);
                return;
            }
        }

        // stopped by `stop`, which waits for this without the GIL
        if (keeps_thread_state &&
            !unloading().load(std::memory_order_acquire) &&
            interpreter_is_live())
        {
            const PyGILState_STATE gstate = PyGILState_Ensure();
            PyGILState_Release(kept);
            PyGILState_Release(gstate);
        }
    }

    /*!
     * @brief Whether `thread` no longer runs, without having reported so.
     *
     * Only known on Windows, where an exiting process kills its other threads
     * before destroying the module's static objects.
     */
    static bool has_exited(std::thread &thread)
    {
#ifdef _WIN32
        return WaitForSingleObject(thread.native_handle(), 0) == WAIT_OBJECT_0;
#else
        static_cast<void>(thread);
        return false;
#endif
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
    bool enabled_ = true;
    bool stopping_ = false;
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

/*!
 * @brief Start `dpctl`'s thread if it is needed and has not been stopped.
 */
inline void start_watcher_if_enabled()
{
    deferred_release_watcher().start_if_enabled();
}

} // namespace detail
} // namespace dpctl

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
    }
#if defined(__GLIBCXX__)
    catch (abi::__forced_unwind &)
    {
        // the interpreter ending this thread, which must not be stopped
        throw;
    }
#endif
    catch (...)
    {
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
 * @brief Stop running deferred releases, waiting a bounded time for the thread.
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

/*!
 * @brief Make deferred releases work again in the child of a `fork`.
 */
inline void reset_deferred_releases_after_fork()
{
    try {
        dpctl::detail::deferred_release_watcher().reset_after_fork();
    } catch (...) {
        // no exception may escape into the calling Cython code
    }
}
