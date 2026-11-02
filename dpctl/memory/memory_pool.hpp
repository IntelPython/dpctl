//===--- memory_pool.hpp                                     -------------===//
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
/// This file declares a cache of USM allocations for one SYCL context, device
/// and USM kind.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>
#include <functional>
#include <mutex>
#include <set>
#include <sycl/sycl.hpp>

namespace dpctl
{
namespace memory
{

class MemoryPool
{
public:
    /*! @brief A part of a USM allocation made by the pool */
    struct Block
    {
        char *ptr;
        std::size_t size;
        bool free;
        bool small;
        // neighbours within the same USM allocation
        Block *prev;
        Block *next;
    };

    MemoryPool(const sycl::context &ctx,
               const sycl::device &dev,
               sycl::usm::alloc kind)
        : ctx_(ctx), dev_(dev), kind_(kind)
    {
    }

    MemoryPool(const MemoryPool &) = delete;
    MemoryPool &operator=(const MemoryPool &) = delete;

    /*! @brief Returns a block of at least `nbytes`, or nullptr if it could
     * not be allocated.
     */
    Block *allocate(std::size_t nbytes) noexcept;

    /*! @brief Returns a block from `allocate` to the pool. No work may be
     * pending on it.
     */
    void release(Block *block) noexcept;

    /*! @brief Frees the USM allocations that have no block in use */
    void empty_cache() noexcept;

private:
    struct BlockLess
    {
        bool operator()(const Block *a, const Block *b) const
        {
            if (a->size != b->size) {
                return a->size < b->size;
            }
            return std::less<char *>()(a->ptr, b->ptr);
        }
    };
    using FreeBlocks = std::set<Block *, BlockLess>;

    sycl::context ctx_;
    sycl::device dev_;
    sycl::usm::alloc kind_;

    std::mutex mutex_;
    FreeBlocks free_small_;
    FreeBlocks free_large_;

    FreeBlocks &free_blocks(bool small)
    {
        return small ? free_small_ : free_large_;
    }
    Block *take_free(std::size_t size, bool small);
    void split(Block *block, std::size_t size);
    char *sycl_malloc(std::size_t nbytes) noexcept;
    void sycl_free(void *ptr) noexcept;
};

class USMPoolDeleter
{
public:
    USMPoolDeleter(MemoryPool *pool, MemoryPool::Block *block)
        : pool_(pool), block_(block)
    {
    }

    void operator()(void *) const noexcept { pool_->release(block_); }

private:
    MemoryPool *pool_;
    MemoryPool::Block *block_;
};

} // namespace memory
} // namespace dpctl
