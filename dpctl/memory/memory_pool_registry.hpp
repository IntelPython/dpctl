//===--- memory_pool_registry.hpp                                 --------===//
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
/// This file declares the registry of memory pools, one per SYCL context,
/// device and USM kind, and the accounting shared by all of them.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>
#include <sycl/sycl.hpp>

#include "memory_pool.hpp"

namespace dpctl
{
namespace memory
{

class MemoryPoolRegistry
{
public:
    /*! @brief Returns the pool for the context, device and USM kind. Host
     * allocations share one pool per context. Pools are never destroyed.
     */
    static MemoryPool *get_pool(const sycl::context &ctx,
                                const sycl::device &dev,
                                sycl::usm::alloc kind);

    /*! @brief Frees the blocks cached by all pools */
    static void empty_cache() noexcept;

    /*! @brief Empties the caches and stops caching for good */
    static void shutdown() noexcept;

    /*! @brief Total bytes the pools may cache. Zero disables caching. Only
     * USM allocations with no block in use can be freed to stay within it.
     */
    static std::size_t get_cache_limit() noexcept;
    /*! @brief Sets the cache limit, emptying the caches if they exceed it */
    static void set_cache_limit(std::size_t nbytes) noexcept;

    /*! @brief Bytes in blocks handed out by the pools and not released */
    static std::size_t used_bytes() noexcept;
    /*! @brief Bytes in free blocks of the pools' USM allocations */
    static std::size_t cached_bytes() noexcept;

    /*! @brief Whether the cached bytes exceed the limit */
    static bool cache_exceeded() noexcept;
    static void add_cached(std::size_t nbytes) noexcept;
    static void remove_cached(std::size_t nbytes) noexcept;
    static void add_used(std::size_t nbytes) noexcept;
    static void remove_used(std::size_t nbytes) noexcept;
};

} // namespace memory
} // namespace dpctl
