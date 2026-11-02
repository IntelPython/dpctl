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
#include <map>
#include <mutex>
#include <sycl/sycl.hpp>
#include <vector>

namespace dpctl
{
namespace memory
{

/*! @brief Size of the block allocated for a request of `nbytes` */
std::size_t pool_block_size(std::size_t nbytes);

class MemoryPool
{
public:
    MemoryPool(const sycl::context &ctx,
               const sycl::device &dev,
               sycl::usm::alloc kind)
        : ctx_(ctx), dev_(dev), kind_(kind)
    {
    }

    MemoryPool(const MemoryPool &) = delete;
    MemoryPool &operator=(const MemoryPool &) = delete;

    /*! @brief Returns a block of at least `nbytes`, or nullptr if it could
     * not be allocated. Stores the size of the block in `block_size`.
     */
    void *allocate(std::size_t nbytes, std::size_t &block_size);

    /*! @brief Caches a block returned by `allocate`, or frees it if the cache
     * is full. The block must have no work pending on it.
     */
    void release(void *ptr, std::size_t block_size) noexcept;

    /*! @brief Frees the cached blocks */
    void empty_cache() noexcept;

private:
    sycl::context ctx_;
    sycl::device dev_;
    sycl::usm::alloc kind_;

    std::mutex mutex_;
    // block size -> cached blocks of that size, no entry is empty
    std::map<std::size_t, std::vector<void *>> cached_;

    void *take_cached(std::size_t block_size, std::size_t &found_size);
    void *sycl_malloc(std::size_t nbytes) noexcept;
    void sycl_free(void *ptr) noexcept;
};

class USMPoolDeleter
{
public:
    USMPoolDeleter(MemoryPool *pool, std::size_t block_size)
        : pool_(pool), block_size_(block_size)
    {
    }

    void operator()(void *ptr) const noexcept
    {
        pool_->release(ptr, block_size_);
    }

private:
    MemoryPool *pool_;
    std::size_t block_size_;
};

} // namespace memory
} // namespace dpctl
