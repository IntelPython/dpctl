//===--- memory_pool.cpp                                     -------------===//
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
//===---------------------------------------------------------------------===//
///
/// \file
/// This file implements a memory pool for caching USM allocations with
/// shared_ptr deleters.
///
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "memory_pool.hpp"

namespace dpctl
{
namespace memory
{

MemoryPool::~MemoryPool()
{
    for (void *ptr : sycl_allocations_) {
        try {
            ::sycl::free(ptr, ctx_);
        } catch (...) {
            // Ignore exceptions during cleanup
        }
    }
}

void *MemoryPool::allocate_from_sycl(std::size_t size)
{
    void *ptr = nullptr;
    switch (type_) {
    case sycl::usm::alloc::host:
        ptr = ::sycl::malloc_host(size, ctx_);
        break;
    case sycl::usm::alloc::device:
        ptr = ::sycl::malloc_device(size, dev_, ctx_);
        break;
    case sycl::usm::alloc::shared:
        ptr = ::sycl::malloc_shared(size, dev_, ctx_);
        break;
    default:
        throw std::invalid_argument("Invalid USM allocation type");
    }

    if (ptr) {
        sycl_allocations_.push_back(ptr);
    }
    return ptr;
}

std::size_t next_power_of_two(std::size_t n)
{
    if (n <= 0)
        return 0;
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

void *MemoryPool::allocate(std::size_t size)
{
    if (size == 0) {
        return nullptr;
    }

    size = next_power_of_two(size);

    {
        std::lock_guard<std::mutex> lock(pool_mutex_);

        auto it = global_free_blocks_.lower_bound(size);
        if (it != global_free_blocks_.end() && !it->second.empty()) {
            void *ptr = it->second.back();
            it->second.pop_back();
            return ptr;
        }
    }

    void *ptr = nullptr;
    try {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        ptr = allocate_from_sycl(size);
    } catch (const sycl::exception &e) {
        ptr = nullptr;
    }

    if (!ptr) {
        empty_cache();

        try {
            std::lock_guard<std::mutex> lock(pool_mutex_);
            ptr = allocate_from_sycl(size);
        } catch (const sycl::exception &e) {
            throw std::bad_alloc();
        }

        if (!ptr) {
            throw std::bad_alloc();
        }
    }

    return ptr;
}

void MemoryPool::free(void *ptr, std::size_t size)
{
    if (!ptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(pool_mutex_);
    global_free_blocks_[size].push_back(ptr);
}

void MemoryPool::empty_cache()
{
    std::lock_guard<std::mutex> lock(pool_mutex_);

    for (auto &[block_size, ptr_list] : global_free_blocks_) {
        for (void *ptr : ptr_list) {
            sycl::free(ptr, ctx_);

            auto raw_it = std::find(sycl_allocations_.begin(),
                                    sycl_allocations_.end(), ptr);
            if (raw_it != sycl_allocations_.end()) {
                sycl_allocations_.erase(raw_it);
            }
        }
        ptr_list.clear();
    }
}

} // namespace memory
} // namespace dpctl
