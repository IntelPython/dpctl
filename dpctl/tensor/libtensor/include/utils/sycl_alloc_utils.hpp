//===-- sycl_alloc_utils.cpp - Allocation utilities ---*-C++-*- ----------====//
//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2025 Intel Corporation
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
/// This file defines CIndexer_array, and CIndexer_vector classes, as well
/// iteration space simplifiers.
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>     // for std::size_t
#include <exception>   // for std::exception
#include <iostream>    // for std::cerr
#include <memory>      // for std::unique_ptr
#include <stdexcept>   // for std::runtime_error
#include <type_traits> // for std::true_type, std::false_type
#include <utility>     // for std::move
#include <vector>

#include "sycl/sycl.hpp"

namespace dpctl
{
namespace tensor
{
namespace alloc_utils
{

template <typename T>
class usm_host_allocator : public sycl::usm_allocator<T, sycl::usm::alloc::host>
{
public:
    using baseT = sycl::usm_allocator<T, sycl::usm::alloc::host>;
    using baseT::baseT;

    template <typename U> struct rebind
    {
        typedef usm_host_allocator<U> other;
    };

    void deallocate(T *ptr, std::size_t n)
    {
        try {
            baseT::deallocate(ptr, n);
        } catch (const std::exception &e) {
            std::cerr
                << "Exception caught in `usm_host_allocator::deallocate`: "
                << e.what() << std::endl;
        }
    }
};

template <typename T>
void sycl_free_noexcept(T *ptr, const sycl::context &ctx) noexcept
{
    try {
        sycl::free(ptr, ctx);
    } catch (const std::exception &e) {
        std::cerr << "Call to sycl::free caught exception: " << e.what()
                  << std::endl;
    }
}

template <typename T>
void sycl_free_noexcept(T *ptr, const sycl::queue &q) noexcept
{
    sycl_free_noexcept(ptr, q.get_context());
}

class USMDeleter
{
private:
    sycl::context ctx_;

public:
    USMDeleter(const sycl::queue &q) : ctx_(q.get_context()) {}
    USMDeleter(const sycl::context &ctx) : ctx_(ctx) {}

    template <typename T> void operator()(T *ptr) const
    {
        sycl_free_noexcept(ptr, ctx_);
    }
};

template <typename T>
std::unique_ptr<T, USMDeleter>
smart_malloc(std::size_t count,
             const sycl::queue &q,
             sycl::usm::alloc kind,
             const sycl::property_list &propList = {})
{
    T *ptr = sycl::malloc<T>(count, q, kind, propList);
    if (nullptr == ptr) {
        throw std::runtime_error("Unable to allocate device_memory");
    }

    auto usm_deleter = USMDeleter(q);
    return std::unique_ptr<T, USMDeleter>(ptr, usm_deleter);
}

template <typename T>
std::unique_ptr<T, USMDeleter>
smart_malloc_device(std::size_t count,
                    const sycl::queue &q,
                    const sycl::property_list &propList = {})
{
    return smart_malloc<T>(count, q, sycl::usm::alloc::device, propList);
}

template <typename T>
std::unique_ptr<T, USMDeleter>
smart_malloc_shared(std::size_t count,
                    const sycl::queue &q,
                    const sycl::property_list &propList = {})
{
    return smart_malloc<T>(count, q, sycl::usm::alloc::shared, propList);
}

template <typename T>
std::unique_ptr<T, USMDeleter>
smart_malloc_host(std::size_t count,
                  const sycl::queue &q,
                  const sycl::property_list &propList = {})
{
    return smart_malloc<T>(count, q, sycl::usm::alloc::host, propList);
}

namespace detail
{
template <typename T> struct valid_smart_ptr : public std::false_type
{
};

template <typename ValT, typename DelT>
struct valid_smart_ptr<std::unique_ptr<ValT, DelT> &>
    : public std::is_same<DelT, USMDeleter>
{
};

template <typename ValT, typename DelT>
struct valid_smart_ptr<std::unique_ptr<ValT, DelT>>
    : public std::is_same<DelT, USMDeleter>
{
};

// base case
template <typename... Rest> struct all_valid_smart_ptrs
{
    static constexpr bool value = true;
};

template <typename Arg, typename... RestArgs>
struct all_valid_smart_ptrs<Arg, RestArgs...>
{
    static constexpr bool value = valid_smart_ptr<Arg>::value &&
                                  (all_valid_smart_ptrs<RestArgs...>::value);
};
} // end of namespace detail

/*! @brief Submit host_task and transfer ownership from smart pointers to it */
template <typename... UniquePtrTs>
sycl::event async_smart_free(sycl::queue &exec_q,
                             const std::vector<sycl::event> &depends,
                             UniquePtrTs &&...unique_pointers)
{
    constexpr std::size_t n = sizeof...(UniquePtrTs);
    static_assert(
        n > 0, "async_smart_free requires at least one smart pointer argument");

    static_assert(
        detail::all_valid_smart_ptrs<UniquePtrTs...>::value,
        "async_smart_free requires unique_ptr created with smart_malloc");

    std::vector<void *> ptrs;
    ptrs.reserve(n);
    (ptrs.push_back(reinterpret_cast<void *>(unique_pointers.get())), ...);

    std::vector<USMDeleter> dels;
    dels.reserve(n);
    (dels.emplace_back(unique_pointers.get_deleter()), ...);

    sycl::event ht_e = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        cgh.host_task([ptrs = std::move(ptrs), dels = std::move(dels)]() {
            for (std::size_t i = 0; i < ptrs.size(); ++i) {
                dels[i](ptrs[i]);
            }
        });
    });

    // Upon successful submission of host_task, USM allocations are owned
    // by the host_task. Release smart pointer ownership to avoid double
    // deallocation
    (unique_pointers.release(), ...);

    return ht_e;
}

} // end of namespace alloc_utils
} // end of namespace tensor
} // end of namespace dpctl
