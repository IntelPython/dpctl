//===--- dpctl_acpp_host_task.hpp - AdaptiveCpp host_task emulation -------===//
//
//                      Data Parallel Control (dpctl)
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a fixed-size thread pool to emulate SYCL host_task
/// functionality. It includes Windows-specific teardown logic mirroring
/// Intel LLVM's thread pool to prevent Loader Lock deadlocks during DLL unload.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/sycl.hpp>

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

namespace dpctl
{
namespace detail
{

class AcppHostTaskPool
{
public:
    static AcppHostTaskPool &get()
    {
        static AcppHostTaskPool instance(4);
        return instance;
    }

    void submit(sycl::event e, std::function<void()> task)
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            tasks_.emplace(std::move(e), std::move(task));
        }
        condition_.notify_one();
    }

    AcppHostTaskPool(const AcppHostTaskPool &) = delete;
    AcppHostTaskPool &operator=(const AcppHostTaskPool &) = delete;

private:
    AcppHostTaskPool(std::size_t num_threads) : stop_(false)
    {
        for (std::size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::pair<sycl::event, std::function<void()>> item;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex_);
                        this->condition_.wait(lock, [this] {
                            return this->stop_ || !this->tasks_.empty();
                        });

                        if (this->stop_ && this->tasks_.empty()) {
#ifdef _WIN32
                            // report to main thread we are about to die
                            {
                                std::lock_guard<std::mutex> exit_lk(
                                    this->win_exit_mutex_);
                                this->win_exit_count_++;
                            }
                            this->win_exit_cv_.notify_one();
#endif
                            return;
                        }

                        item = std::move(this->tasks_.front());
                        this->tasks_.pop();
                    }

                    item.first.wait();
                    item.second();
                }
            });
        }
    }

    ~AcppHostTaskPool()
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();

#ifdef _WIN32
        // wait for workers to check in before detach
        {
            std::unique_lock<std::mutex> lock(win_exit_mutex_);
            win_exit_cv_.wait(
                lock, [this] { return win_exit_count_ == workers_.size(); });
        }
        for (std::thread &worker : workers_) {
            if (worker.joinable()) {
                worker.detach();
            }
        }
#else
        for (std::thread &worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
#endif
    }

    std::vector<std::thread> workers_;
    std::queue<std::pair<sycl::event, std::function<void()>>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_;

#ifdef _WIN32
    std::mutex win_exit_mutex_;
    std::condition_variable win_exit_cv_;
    std::size_t win_exit_count_ = 0;
#endif
};

} // namespace detail
} // namespace dpctl
