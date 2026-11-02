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
#include <mutex>
#include <new>
#include <sycl/sycl.hpp>
#include <vector>

#include "memory_pool.hpp"
#include "memory_pool_registry.hpp"

namespace dpctl
{
namespace memory
{

namespace
{

constexpr std::size_t block_granularity = 512;
// requests up to this size come from small allocations, and blocks of large
// allocations are only split if more than this would be left over
constexpr std::size_t small_block_limit = std::size_t(1) << 20;
constexpr std::size_t small_allocation_size = std::size_t(2) << 20;

std::size_t round_up(std::size_t n, std::size_t step)
{
    const std::size_t rem = n % step;
    if (rem == 0) {
        return n;
    }
    if (n > std::numeric_limits<std::size_t>::max() - (step - rem)) {
        return 0;
    }
    return n + (step - rem);
}

} // namespace

MemoryPool::Block *MemoryPool::allocate(std::size_t nbytes) noexcept
{
    if (nbytes == 0) {
        return nullptr;
    }

    const bool caching = MemoryPoolRegistry::get_cache_limit() > 0;
    std::size_t size = nbytes;
    bool small = false;

    if (caching) {
        size = round_up(nbytes, block_granularity);
        if (size == 0) {
            return nullptr;
        }
        small = size <= small_block_limit;

        std::lock_guard<std::mutex> lock(mutex_);
        if (Block *block = take_free(size, small)) {
            split(block, size);
            MemoryPoolRegistry::add_used(block->size);
            return block;
        }
    }

    const std::size_t alloc_size = small ? small_allocation_size : size;
    char *ptr = sycl_malloc(alloc_size);
    if (!ptr) {
        return nullptr;
    }
    Block *block = new (std::nothrow)
        Block{ptr, alloc_size, false, small, nullptr, nullptr};
    if (!block) {
        sycl_free(ptr);
        return nullptr;
    }
    if (caching) {
        std::lock_guard<std::mutex> lock(mutex_);
        split(block, size);
    }
    MemoryPoolRegistry::add_used(block->size);
    return block;
}

void MemoryPool::release(Block *block) noexcept
{
    void *to_free = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        FreeBlocks &free_set = free_blocks(block->small);

        MemoryPoolRegistry::remove_used(block->size);
        MemoryPoolRegistry::add_cached(block->size);
        block->free = true;

        if (Block *prev = block->prev; prev && prev->free) {
            free_set.erase(prev);
            prev->size += block->size;
            prev->next = block->next;
            if (block->next) {
                block->next->prev = prev;
            }
            delete block;
            block = prev;
        }
        if (Block *next = block->next; next && next->free) {
            free_set.erase(next);
            block->size += next->size;
            block->next = next->next;
            if (next->next) {
                next->next->prev = block;
            }
            delete next;
        }

        const bool whole = !block->prev && !block->next;
        if (whole && MemoryPoolRegistry::cache_exceeded()) {
            MemoryPoolRegistry::remove_cached(block->size);
            to_free = block->ptr;
            delete block;
        }
        else {
            try {
                free_set.insert(block);
            } catch (...) {
                // leaked, never merged with its neighbours again
                block->free = false;
                MemoryPoolRegistry::remove_cached(block->size);
            }
        }
    }
    if (to_free) {
        sycl_free(to_free);
    }
}

void MemoryPool::empty_cache() noexcept
{
    std::vector<void *> to_free;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (FreeBlocks *free_set : {&free_small_, &free_large_}) {
            for (auto it = free_set->begin(); it != free_set->end();) {
                Block *block = *it;
                if (block->prev || block->next) {
                    ++it;
                    continue;
                }
                try {
                    to_free.push_back(block->ptr);
                } catch (...) {
                    break;
                }
                MemoryPoolRegistry::remove_cached(block->size);
                it = free_set->erase(it);
                delete block;
            }
        }
    }
    for (void *ptr : to_free) {
        sycl_free(ptr);
    }
}

// expects mutex_ to be held
MemoryPool::Block *MemoryPool::take_free(std::size_t size, bool small)
{
    FreeBlocks &free_set = free_blocks(small);
    Block probe{nullptr, size, true, small, nullptr, nullptr};
    auto it = free_set.lower_bound(&probe);
    if (it == free_set.end()) {
        return nullptr;
    }
    Block *block = *it;
    free_set.erase(it);
    block->free = false;
    MemoryPoolRegistry::remove_cached(block->size);
    return block;
}

// expects mutex_ to be held; leaves `block` whole if splitting fails
void MemoryPool::split(Block *block, std::size_t size)
{
    const std::size_t rem = block->size - size;
    const bool worth_it =
        block->small ? rem >= block_granularity : rem > small_block_limit;
    if (!worth_it) {
        return;
    }
    Block *rest = new (std::nothrow)
        Block{block->ptr + size, rem, true, block->small, block, block->next};
    if (!rest) {
        return;
    }
    try {
        free_blocks(block->small).insert(rest);
    } catch (...) {
        delete rest;
        return;
    }
    if (block->next) {
        block->next->prev = rest;
    }
    block->next = rest;
    block->size = size;
    MemoryPoolRegistry::add_cached(rem);
}

char *MemoryPool::sycl_malloc(std::size_t nbytes) noexcept
{
    try {
        switch (kind_) {
        case sycl::usm::alloc::host:
            return static_cast<char *>(sycl::malloc_host(nbytes, ctx_));
        case sycl::usm::alloc::device:
            return static_cast<char *>(sycl::malloc_device(nbytes, dev_, ctx_));
        case sycl::usm::alloc::shared:
            return static_cast<char *>(sycl::malloc_shared(nbytes, dev_, ctx_));
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
