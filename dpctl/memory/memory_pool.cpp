//===--- memory_pool.cpp                                     -------------===//
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
//===---------------------------------------------------------------------===//
///
/// \file
/// This file implements a cache of USM allocations for one SYCL context,
/// device and USM kind.
///
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

#include "memory_pool.hpp"
#include "memory_pool_registry.hpp"

namespace dpctl
{
namespace memory
{

namespace
{

constexpr std::size_t small_block_granularity = 512;
constexpr std::size_t small_block_limit = std::size_t(1) << 20;
// steps between consecutive powers of two for large blocks
constexpr std::size_t large_block_divisions = 8;

std::size_t round_up(std::size_t n, std::size_t step)
{
    const std::size_t rem = n % step;
    if (rem == 0) {
        return n;
    }
    if (n > std::numeric_limits<std::size_t>::max() - (step - rem)) {
        return n;
    }
    return n + (step - rem);
}

// largest block this pool hands out for a request of block size `n`
std::size_t max_reuse_size(std::size_t n)
{
    const std::size_t slack = n / 4;
    if (n > std::numeric_limits<std::size_t>::max() - slack) {
        return std::numeric_limits<std::size_t>::max();
    }
    return n + slack;
}

} // namespace

std::size_t pool_block_size(std::size_t nbytes)
{
    if (nbytes <= small_block_limit) {
        return round_up(nbytes, small_block_granularity);
    }
    std::size_t pow2 = small_block_limit;
    while (pow2 <= nbytes / 2) {
        pow2 <<= 1;
    }
    return round_up(nbytes, pow2 / large_block_divisions);
}

void *MemoryPool::allocate(std::size_t nbytes, std::size_t &block_size)
{
    const bool caching = MemoryPoolRegistry::get_cache_limit() > 0;
    const std::size_t size = caching ? pool_block_size(nbytes) : nbytes;

    if (caching) {
        std::size_t found_size = 0;
        void *ptr = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ptr = take_cached(size, found_size);
        }
        if (ptr) {
            MemoryPoolRegistry::uncache(found_size);
            MemoryPoolRegistry::add_used(found_size);
            block_size = found_size;
            return ptr;
        }
    }

    void *ptr = sycl_malloc(size);
    if (ptr) {
        MemoryPoolRegistry::add_used(size);
        block_size = size;
    }
    return ptr;
}

void MemoryPool::release(void *ptr, std::size_t block_size) noexcept
{
    if (!ptr) {
        return;
    }
    MemoryPoolRegistry::remove_used(block_size);

    if (MemoryPoolRegistry::try_cache(block_size)) {
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            cached_[block_size].push_back(ptr);
            return;
        } catch (...) {
            MemoryPoolRegistry::uncache(block_size);
        }
    }
    sycl_free(ptr);
}

void MemoryPool::empty_cache() noexcept
{
    std::map<std::size_t, std::vector<void *>> blocks;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        blocks.swap(cached_);
    }
    for (auto &[block_size, ptrs] : blocks) {
        MemoryPoolRegistry::uncache(block_size * ptrs.size());
        for (void *ptr : ptrs) {
            sycl_free(ptr);
        }
    }
}

void *MemoryPool::take_cached(std::size_t block_size, std::size_t &found_size)
{
    auto it = cached_.lower_bound(block_size);
    if (it == cached_.end() || it->first > max_reuse_size(block_size)) {
        return nullptr;
    }
    void *ptr = it->second.back();
    it->second.pop_back();
    found_size = it->first;
    if (it->second.empty()) {
        cached_.erase(it);
    }
    return ptr;
}

void *MemoryPool::sycl_malloc(std::size_t nbytes) noexcept
{
    try {
        switch (kind_) {
        case sycl::usm::alloc::host:
            return sycl::malloc_host(nbytes, ctx_);
        case sycl::usm::alloc::device:
            return sycl::malloc_device(nbytes, dev_, ctx_);
        case sycl::usm::alloc::shared:
            return sycl::malloc_shared(nbytes, dev_, ctx_);
        default:
            return nullptr;
        }
    } catch (...) {
        return nullptr;
    }
}

void MemoryPool::sycl_free(void *ptr) noexcept
{
    try {
        sycl::free(ptr, ctx_);
    } catch (const std::exception &e) {
        std::cout << "Call to sycl::free caught an exception: " << e.what()
                  << std::endl;
    }
}

} // namespace memory
} // namespace dpctl
