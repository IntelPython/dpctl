//===--- deferred_releases.hpp - keeps owners alive during offload --------===//
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
/// This file implements a list of releases of the objects used by offloaded
/// tasks, each deferred until the task using them has completed, and run by a
/// thread that `dpctl` owns.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#if defined(__GLIBCXX__)
#include <cxxabi.h>
#endif

namespace dpctl
{
namespace detail
{

/*!
 * @brief Releases of objects used by offloaded tasks, held until they complete.
 *
 * A release runs on the thread `dpctl` keeps for looking over the list, or on a
 * thread that calls `drain`. The events an item is held for say when the
 * objects stopped being used, not when they were released, so nothing should
 * wait for a release to happen.
 *
 * An item is released only once its events report complete, so nothing is freed
 * while a task may still be using it. Whether to keep releasing as the
 * interpreter shuts down is up to the caller: nothing here checks.
 *
 * Neither the list nor the items left on it are ever destroyed, so the events
 * and allocations they hold are not released during static teardown, when the
 * SYCL runtime may be gone. What is still held when a process ends is left for
 * it to reclaim.
 *
 * Deferring runs no release, so a caller may hold whatever locks it likes.
 * Whoever runs one has to be able to run Python code.
 *
 * A release drops references, but what that reclaims is up to the interpreter.
 * On a free-threaded build a reference dropped by a thread that does not own
 * the object need not run its destructor at once, so an allocation the object
 * holds can be reclaimed later than the reference drop.
 */
class DeferredReleases
{
public:
    using clock = std::chrono::steady_clock;

    /*!
     * @brief Defer `release` until every event in `depends` has completed.
     *
     * `release` must own what it releases, so move USM `shared_ptr` copies or
     * `PyObject *` references into it.
     */
    void defer(std::vector<sycl::event> depends, std::function<void()> release)
    {
        push(Item{std::move(depends), std::move(release)});

        if (on_defer_) {
            try {
                on_defer_();
            } catch (...) {
                // the release is already deferred, so a failure is not reported
            }
        }
    }

    /*!
     * @brief Move the releases that have come due onto `due`.
     *
     * Takes them off the list, so the caller is responsible for running them.
     * The events are queried without holding the list's lock, so a long list
     * does not hold up `defer`.
     *
     * @return Whether anything was added to `due`.
     */
    bool collect_due(std::vector<std::function<void()>> &due)
    {
        auto items = std::make_unique<std::vector<Item>>();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            items->swap(deferred_);
        }

        bool any = false;
        bool any_left = false;
        try {
            const auto due_begin = std::partition(
                items->begin(), items->end(),
                [](const Item &item) { return !is_complete(item); });
            any = due_begin != items->end();

            due.reserve(due.size() + (items->end() - due_begin));
            for (auto it = due_begin; it != items->end(); ++it) {
                due.push_back(std::move(it->release));
            }
            items->erase(due_begin, items->end());

            std::lock_guard<std::mutex> lock(mutex_);
            // add what was deferred during the scan
            items->insert(items->end(),
                          std::make_move_iterator(deferred_.begin()),
                          std::make_move_iterator(deferred_.end()));
            deferred_.swap(*items);
            any_left = !deferred_.empty();
        } catch (...) {
            // leaked rather than destroyed, as a task may still be using what
            // the items hold
            static_cast<void>(items.release());
            throw;
        }

        // a waiter may have seen the list empty while it was taken off
        if (any_left) {
            deferred_or_stopped_.notify_all();
        }

        return any;
    }

    /*!
     * @brief Run and drop the releases collected in `due`.
     *
     * Expects the caller to have a Python thread state, as a release runs
     * Python code, which is free to defer more releases and to call back in
     * here. A release that does call back in runs nothing itself, so releases
     * do not nest on one thread.
     *
     * If `until_stopped`, the releases not yet run once `stop_waiting` has
     * been called are put back on the list instead.
     */
    void run_collected(std::vector<std::function<void()>> &due,
                       bool until_stopped = false)
    {
        bool &running = running_here();
        const bool was_running = running;
        running = true;

        for (std::size_t i = 0; i < due.size(); ++i) {
            if (until_stopped && stop_waiting_.load(std::memory_order_acquire))
            {
                put_back(due, i);
                break;
            }

            try {
                due[i]();
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
                // the rest must still run
            }
        }
        due.clear();

        running = was_running;
    }

    /*!
     * @brief Put the releases in `due` from `first` on back on the list.
     *
     * They are held for no events, so they run on the next pass.
     */
    void put_back(std::vector<std::function<void()>> &due,
                  std::size_t first = 0)
    {
        for (std::size_t i = first; i < due.size(); ++i) {
            push(Item{{}, std::move(due[i])});
        }
        due.clear();
    }

    /*!
     * @brief Run every release that has come due, on the calling thread.
     *
     * Expects the caller to have a Python thread state, and does nothing if it
     * is itself called from a release.
     *
     * Makes one pass, leaving what a release defers, what a drain called from
     * inside one skipped, and what another pass has taken off the list at the
     * time, to the thread looking over the list.
     *
     * @return Whether there was anything to run.
     */
    bool drain()
    {
        if (running_here()) {
            return false;
        }

        std::vector<std::function<void()>> due;
        if (!collect_due(due)) {
            return false;
        }

        run_collected(due);

        return true;
    }

    /*!
     * @brief Wait for something that may have come due.
     *
     * Blocks while the list is empty, as nothing can come due until something
     * is deferred, and otherwise for at most `poll_interval`: an item comes due
     * on its own, with nothing to announce it.
     *
     * @return Whether waiting is still in order, and false once `stop_waiting`
     * has been called.
     */
    bool wait_for_due(clock::duration poll_interval)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        if (deferred_.empty()) {
            deferred_or_stopped_.wait(lock, [this]() {
                return stop_waiting_.load(std::memory_order_relaxed) ||
                       !deferred_.empty();
            });
        }
        else {
            deferred_or_stopped_.wait_for(lock, poll_interval, [this]() {
                return stop_waiting_.load(std::memory_order_relaxed);
            });
        }

        return !stop_waiting_.load(std::memory_order_relaxed);
    }

    /*!
     * @brief End every wait for something to come due, and refuse further ones.
     */
    void stop_waiting()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_waiting_.store(true, std::memory_order_release);
        }

        deferred_or_stopped_.notify_all();
    }

    /*!
     * @brief Allow waiting again, after a `stop_waiting`.
     */
    void resume_waiting()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_waiting_.store(false, std::memory_order_release);
    }

    /*!
     * @brief Make the list usable in the child of a `fork`.
     *
     * Replaces the lock and condition variable, which a thread that does not
     * exist in the child may have held, and leaks the parent's items, whose
     * events belong to the parent.
     */
    void reset_after_fork()
    {
        new (&mutex_) std::mutex();
        new (&deferred_or_stopped_) std::condition_variable();
        static_cast<void>(new std::vector<Item>(std::move(deferred_)));
        deferred_.clear();
        stop_waiting_.store(false, std::memory_order_release);
    }

    DeferredReleases(const DeferredReleases &) = delete;
    DeferredReleases &operator=(const DeferredReleases &) = delete;
    ~DeferredReleases() = delete;

private:
    struct Item
    {
        std::vector<sycl::event> depends;
        std::function<void()> release;
    };

    /*!
     * @brief The only place a list is created.
     *
     * `dpctl._sycl_queue` defines it and hands the list out through
     * `DeferredReleases_Get`. Only the name is befriended, so a translation
     * unit that declares it gets a list of its own, which nothing drains.
     */
    friend DeferredReleases &local_deferred_releases();

    /*!
     * @brief Create a list that calls `on_defer` after every `defer`.
     */
    explicit DeferredReleases(void (*on_defer)()) : on_defer_(on_defer) {}

    /*!
     * @brief Add `item` to the list, waking a waiter if it was empty.
     */
    void push(Item item)
    {
        bool was_empty = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            was_empty = deferred_.empty();
            deferred_.push_back(std::move(item));
        }

        // only an empty list is waited on without a timeout
        if (was_empty) {
            deferred_or_stopped_.notify_all();
        }
    }

    /*!
     * @brief Whether a release is running on this thread.
     */
    static bool &running_here()
    {
        static thread_local bool running = false;
        return running;
    }

    /*!
     * @brief Whether every event in `item.depends` has completed.
     *
     * An event whose status cannot be read is reported as not complete, as it
     * is no evidence that the task is done with what the release would free.
     * The item is looked at again by every later pass, and is leaked if the
     * status never becomes readable.
     */
    static bool is_complete(const Item &item)
    {
        static constexpr auto complete =
            sycl::info::event_command_status::complete;

        try {
            for (const auto &e : item.depends) {
                if (e.get_info<sycl::info::event::command_execution_status>() !=
                    complete)
                {
                    return false;
                }
            }
        } catch (...) {
            return false;
        }

        return true;
    }

    std::vector<Item> deferred_;
    std::atomic<bool> stop_waiting_{false};
    std::mutex mutex_;
    std::condition_variable deferred_or_stopped_;

    /*!
     * @brief Called after every `defer`.
     *
     * `dpctl` uses it to start its thread on the first release deferred.
     */
    void (*on_defer_)() = nullptr;
};

/*!
 * @brief Name of the kernel submitted by `submit_keep_alive_marker`.
 */
class keep_alive_marker;

/*!
 * @brief An event that gates the release of objects used by work on `q`.
 *
 * Submits an empty kernel after `deps`. On an in-order queue the submission is
 * also ordered after everything already submitted there, so the event covers
 * uses of the objects that the caller never tracked (gating on `deps` alone
 * would release them while that work still runs). On an out-of-order queue it
 * adds nothing over `deps` beyond collapsing them into one event.
 *
 * @return An event that completes once the objects have stopped being used.
 */
inline sycl::event
submit_keep_alive_marker(sycl::queue &q, const std::vector<sycl::event> &deps)
{
    return q.single_task<keep_alive_marker>(deps, []() {});
}

} // namespace detail
} // namespace dpctl
