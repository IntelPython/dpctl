#pragma once
#include <sycl/sycl.hpp>

#include <algorithm>
#include <vector>

namespace
{
bool is_event_complete(const sycl::event &e)
{
    constexpr auto exec_complete = sycl::info::event_command_status::complete;

    const auto status =
        e.get_info<sycl::info::event::command_execution_status>();
    return (status == exec_complete);
}
} // namespace

class SequentialOrder
{
private:
    std::vector<sycl::event> host_task_events;
    std::vector<sycl::event> submitted_events;

    void prune_complete()
    {
        const auto &ht_it =
            std::remove_if(host_task_events.begin(), host_task_events.end(),
                           is_event_complete);
        host_task_events.erase(ht_it, host_task_events.end());

        const auto &sub_it =
            std::remove_if(submitted_events.begin(), submitted_events.end(),
                           is_event_complete);
        submitted_events.erase(sub_it, submitted_events.end());
    }

public:
    SequentialOrder() : host_task_events{}, submitted_events{} {}
    SequentialOrder(size_t n) : host_task_events{}, submitted_events{}
    {
        host_task_events.reserve(n);
        submitted_events.reserve(n);
    }

    SequentialOrder(const SequentialOrder &other)
        : host_task_events(other.host_task_events),
          submitted_events(other.submitted_events)
    {
        prune_complete();
    }
    SequentialOrder(SequentialOrder &&other)
        : host_task_events{}, submitted_events{}
    {
        host_task_events = std::move(other.host_task_events);
        submitted_events = std::move(other.submitted_events);
        prune_complete();
    }

    SequentialOrder &operator=(const SequentialOrder &other)
    {
        host_task_events = other.host_task_events;
        submitted_events = other.submitted_events;

        prune_complete();
        return *this;
    }

    SequentialOrder &operator=(SequentialOrder &&other)
    {
        if (this != &other) {
            host_task_events = std::move(other.host_task_events);
            submitted_events = std::move(other.submitted_events);
            prune_complete();
        }
        return *this;
    }

    size_t get_num_submitted_events() const
    {
        return submitted_events.size();
    }

    const std::vector<sycl::event> &get_host_task_events()
    {
        prune_complete();
        return host_task_events;
    }

    /*
        const std::vector<sycl::event> & get_host_task_events() const {
            return host_task_events;
        }
    */

    size_t get_num_host_task_events() const
    {
        return host_task_events.size();
    }

    const std::vector<sycl::event> &get_submitted_events()
    {
        prune_complete();
        return submitted_events;
    }

    /*
        const std::vector<sycl::event> & get_submitted_events() const {
            return submitted_events;
        }
    */

    void add_to_both_events(const sycl::event &ht_ev,
                            const sycl::event &comp_ev)
    {
        prune_complete();
        if (!is_event_complete(ht_ev))
            host_task_events.push_back(ht_ev);
        if (!is_event_complete(comp_ev))
            submitted_events.push_back(comp_ev);
    }

    void add_vector_to_both_events(const std::vector<sycl::event> &ht_evs,
                                   const std::vector<sycl::event> &comp_evs)
    {
        prune_complete();
        for (const auto &e : ht_evs) {
            if (!is_event_complete(e))
                host_task_events.push_back(e);
        }
        for (const auto &e : comp_evs) {
            if (!is_event_complete(e))
                submitted_events.push_back(e);
        }
    }

    void add_to_host_task_events(const sycl::event &ht_ev)
    {
        prune_complete();
        if (!is_event_complete(ht_ev)) {
            host_task_events.push_back(ht_ev);
        }
    }

    void add_to_submitted_events(const sycl::event &comp_ev)
    {
        prune_complete();
        if (!is_event_complete(comp_ev)) {
            submitted_events.push_back(comp_ev);
        }
    }

    template <std::size_t num>
    void add_list_to_host_task_events(const sycl::event (&ht_events)[num])
    {
        prune_complete();
        for (size_t i = 0; i < num; ++i) {
            const auto &e = ht_events[i];
            if (!is_event_complete(e))
                host_task_events.push_back(e);
        }
    }

    template <std::size_t num>
    void add_list_to_submitted_events(const sycl::event (&comp_events)[num])
    {
        prune_complete();
        for (size_t i = 0; i < num; ++i) {
            const auto &e = comp_events[i];
            if (!is_event_complete(e))
                submitted_events.push_back(e);
        }
    }

    void wait()
    {
        sycl::event::wait(submitted_events);
        sycl::event::wait(host_task_events);
        prune_complete();
    }
};
