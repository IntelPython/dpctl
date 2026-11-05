#pragma once
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <utility>
#include <vector>

namespace
{
inline bool is_event_complete(const sycl::event &e)
{
    static constexpr auto exec_complete =
        sycl::info::event_command_status::complete;

    const auto status =
        e.get_info<sycl::info::event::command_execution_status>();
    return (status == exec_complete);
}
} // namespace

class SequentialOrder
{
private:
    mutable std::mutex mu_events;
    // events that gate the release of objects used by offloaded tasks, such as
    // those returned by `dpctl::utils::keep_args_alive`. They are waited on,
    // but never used as dependencies of later tasks.
    std::vector<sycl::event> cleanup_events;
    // events for the offloaded tasks themselves, used as the dependencies of
    // the tasks that follow them
    std::vector<sycl::event> submitted_events;

    // only called with mu_events held
    void prune_complete_nolock()
    {
        const auto &cl_it = std::remove_if(
            cleanup_events.begin(), cleanup_events.end(), is_event_complete);
        cleanup_events.erase(cl_it, cleanup_events.end());

        const auto &sub_it =
            std::remove_if(submitted_events.begin(), submitted_events.end(),
                           is_event_complete);
        submitted_events.erase(sub_it, submitted_events.end());
    }

public:
    SequentialOrder() : cleanup_events{}, submitted_events{} {}
    SequentialOrder(std::size_t n) : cleanup_events{}, submitted_events{}
    {
        cleanup_events.reserve(n);
        submitted_events.reserve(n);
    }

    SequentialOrder(const SequentialOrder &other)
    {
        std::lock_guard<std::mutex> lock(other.mu_events);
        cleanup_events = other.cleanup_events;
        submitted_events = other.submitted_events;
        prune_complete_nolock();
    }
    SequentialOrder(SequentialOrder &&other)
        : cleanup_events{}, submitted_events{}
    {
        std::lock_guard<std::mutex> lock(other.mu_events);
        cleanup_events = std::move(other.cleanup_events);
        submitted_events = std::move(other.submitted_events);
        prune_complete_nolock();
    }

    SequentialOrder &operator=(const SequentialOrder &other)
    {
        if (this != &other) {
            std::scoped_lock lock(mu_events, other.mu_events);
            cleanup_events = other.cleanup_events;
            submitted_events = other.submitted_events;
            prune_complete_nolock();
        }
        return *this;
    }

    SequentialOrder &operator=(SequentialOrder &&other)
    {
        if (this != &other) {
            std::scoped_lock lock(mu_events, other.mu_events);
            cleanup_events = std::move(other.cleanup_events);
            submitted_events = std::move(other.submitted_events);
            prune_complete_nolock();
        }
        return *this;
    }

    std::size_t get_num_submitted_events() const
    {
        std::lock_guard<std::mutex> lock(mu_events);
        return submitted_events.size();
    }

    // returns a copy to avoid returning a reference that
    // could be modified after the lock is released
    std::vector<sycl::event> get_cleanup_events()
    {
        std::lock_guard<std::mutex> lock(mu_events);
        prune_complete_nolock();
        return cleanup_events;
    }

    std::size_t get_num_cleanup_events() const
    {
        std::lock_guard<std::mutex> lock(mu_events);
        return cleanup_events.size();
    }

    // returns a copy to avoid returning a reference that
    // could be modified after the lock is released
    std::vector<sycl::event> get_submitted_events()
    {
        std::lock_guard<std::mutex> lock(mu_events);
        prune_complete_nolock();
        return submitted_events;
    }

    void add_to_both_events(const sycl::event &cleanup_ev,
                            const sycl::event &comp_ev)
    {
        std::lock_guard<std::mutex> lock(mu_events);
        prune_complete_nolock();
        if (!is_event_complete(cleanup_ev))
            cleanup_events.push_back(cleanup_ev);
        if (!is_event_complete(comp_ev))
            submitted_events.push_back(comp_ev);
    }

    void add_vector_to_both_events(const std::vector<sycl::event> &cleanup_evs,
                                   const std::vector<sycl::event> &comp_evs)
    {
        std::lock_guard<std::mutex> lock(mu_events);
        prune_complete_nolock();
        for (const auto &e : cleanup_evs) {
            if (!is_event_complete(e))
                cleanup_events.push_back(e);
        }
        for (const auto &e : comp_evs) {
            if (!is_event_complete(e))
                submitted_events.push_back(e);
        }
    }

    void add_to_cleanup_events(const sycl::event &cleanup_ev)
    {
        std::lock_guard<std::mutex> lock(mu_events);
        prune_complete_nolock();
        if (!is_event_complete(cleanup_ev)) {
            cleanup_events.push_back(cleanup_ev);
        }
    }

    void add_to_submitted_events(const sycl::event &comp_ev)
    {
        std::lock_guard<std::mutex> lock(mu_events);
        prune_complete_nolock();
        if (!is_event_complete(comp_ev)) {
            submitted_events.push_back(comp_ev);
        }
    }

    template <std::size_t num>
    void add_list_to_cleanup_events(const sycl::event (&cleanup_evs)[num])
    {
        std::lock_guard<std::mutex> lock(mu_events);
        prune_complete_nolock();
        for (std::size_t i = 0; i < num; ++i) {
            const auto &e = cleanup_evs[i];
            if (!is_event_complete(e))
                cleanup_events.push_back(e);
        }
    }

    template <std::size_t num>
    void add_list_to_submitted_events(const sycl::event (&comp_events)[num])
    {
        std::lock_guard<std::mutex> lock(mu_events);
        prune_complete_nolock();
        for (std::size_t i = 0; i < num; ++i) {
            const auto &e = comp_events[i];
            if (!is_event_complete(e))
                submitted_events.push_back(e);
        }
    }

    void wait()
    {
        // snapshot events outside of mutex to avoid
        // calling wait inside mutex
        std::vector<sycl::event> sub_copy;
        std::vector<sycl::event> cl_copy;
        {
            std::lock_guard<std::mutex> lock(mu_events);
            sub_copy = submitted_events;
            cl_copy = cleanup_events;
        }
        sycl::event::wait(sub_copy);
        sycl::event::wait(cl_copy);
        {
            std::lock_guard<std::mutex> lock(mu_events);
            prune_complete_nolock();
        }
    }
};
