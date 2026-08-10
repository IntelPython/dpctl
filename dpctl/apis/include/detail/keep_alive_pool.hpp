//===--- keep_alive_pool.hpp - keeps owners alive during offload ----------===//
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
/// A fixed-size pool of threads that wait on SYCL events and then run a
/// callable for maintaining Python object lifetime during offloaded tasks.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

namespace dpctl
{
namespace detail
{

class KeepAlivePool
{
public:
    /*!
     * @brief Number of waiter threads.
     */
    static constexpr std::size_t num_threads = 4;

    static KeepAlivePool &get()
    {
        // deliberately leaked: workers are detached and hold a bare `this`, so
        // the pool must outlive them
        static KeepAlivePool *instance = new KeepAlivePool();
        return *instance;
    }

    /*!
     * @brief Run `task` once every event in `depends` has completed.
     *
     * `task` must own everything it releases -- move USM `shared_ptr` copies,
     * `sycl::buffer` handles, or `PyObject *` references into it. It runs on a
     * pool thread without the GIL, so it must take the GIL itself if it touches
     * Python.
     */
    void submit(std::vector<sycl::event> depends, std::function<void()> task)
    {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            tasks_.emplace(std::move(depends), std::move(task));
        }
        condition_.notify_one();
    }

    KeepAlivePool(const KeepAlivePool &) = delete;
    KeepAlivePool &operator=(const KeepAlivePool &) = delete;
    ~KeepAlivePool() = delete;

private:
    KeepAlivePool()
    {
        for (std::size_t i = 0; i < num_threads; ++i) {
            std::thread(&KeepAlivePool::run, this).detach();
        }
    }

    void run()
    {
        for (;;) {
            std::pair<std::vector<sycl::event>, std::function<void()>> item;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                condition_.wait(lock, [this] { return !tasks_.empty(); });
                item = std::move(tasks_.front());
                tasks_.pop();
            }

            try {
                sycl::event::wait(item.first);
            } catch (const std::exception &) {
                // run the task anyway: an async error must not strand the
                // task or else it may leak
            }

            try {
                item.second();
            } catch (const std::exception &) {
                // a throwing task must not take down the worker or later
                // tasks will be lost
            }
        }
    }

    std::queue<std::pair<std::vector<sycl::event>, std::function<void()>>>
        tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
};

} // namespace detail
} // namespace dpctl
