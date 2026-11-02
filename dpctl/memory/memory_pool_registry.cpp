//===--- memory_pool_registry.cpp                                 --------===//
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
/// This file implements the registry of memory pools, one per SYCL context,
/// device and USM kind, and the accounting shared by all of them.
///
//===----------------------------------------------------------------------===//

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <sycl/sycl.hpp>
#include <unordered_map>
#include <vector>

#include "memory_pool.hpp"
#include "memory_pool_registry.hpp"

namespace dpctl
{
namespace memory
{

namespace
{

struct PoolKey
{
    sycl::context ctx;
    sycl::device dev;
    sycl::usm::alloc kind;

    bool operator==(const PoolKey &other) const
    {
        return kind == other.kind && ctx == other.ctx &&
               (kind == sycl::usm::alloc::host || dev == other.dev);
    }
};

struct PoolKeyHash
{
    std::size_t operator()(const PoolKey &key) const
    {
        std::size_t h = std::hash<sycl::context>()(key.ctx);
        h = h * 31 + static_cast<std::size_t>(key.kind);
        if (key.kind != sycl::usm::alloc::host) {
            h = h * 31 + std::hash<sycl::device>()(key.dev);
        }
        return h;
    }
};

std::size_t initial_cache_limit()
{
    const char *env = std::getenv("DPCTL_USM_POOL");
    if (env && std::strcmp(env, "0") == 0) {
        return 0;
    }
    return std::numeric_limits<std::size_t>::max();
}

struct RegistryState
{
    std::mutex mutex;
    std::unordered_map<PoolKey, std::unique_ptr<MemoryPool>, PoolKeyHash> pools;

    std::atomic<std::size_t> cache_limit{initial_cache_limit()};
    std::atomic<std::size_t> cached{0};
    std::atomic<std::size_t> used{0};
    std::atomic<bool> shut_down{false};
};

// leaked, so that nothing frees USM or destroys SYCL objects during static
// destruction
RegistryState &get_state()
{
    static RegistryState *state = new RegistryState;
    return *state;
}

std::vector<MemoryPool *> all_pools()
{
    auto &state = get_state();
    std::vector<MemoryPool *> pools;
    std::lock_guard<std::mutex> lock(state.mutex);
    pools.reserve(state.pools.size());
    for (auto &entry : state.pools) {
        pools.push_back(entry.second.get());
    }
    return pools;
}

} // namespace

MemoryPool *MemoryPoolRegistry::get_pool(const sycl::context &ctx,
                                         const sycl::device &dev,
                                         sycl::usm::alloc kind)
{
    auto &state = get_state();
    PoolKey key{ctx, dev, kind};

    std::lock_guard<std::mutex> lock(state.mutex);
    auto &pool = state.pools[key];
    if (!pool) {
        pool = std::make_unique<MemoryPool>(ctx, dev, kind);
    }
    return pool.get();
}

void MemoryPoolRegistry::empty_cache() noexcept
{
    std::vector<MemoryPool *> pools;
    try {
        pools = all_pools();
    } catch (...) {
        return;
    }
    for (MemoryPool *pool : pools) {
        pool->empty_cache();
    }
}

void MemoryPoolRegistry::shutdown() noexcept
{
    auto &state = get_state();
    state.shut_down.store(true);
    state.cache_limit.store(0);
    empty_cache();
}

std::size_t MemoryPoolRegistry::get_cache_limit() noexcept
{
    return get_state().cache_limit.load();
}

void MemoryPoolRegistry::set_cache_limit(std::size_t nbytes) noexcept
{
    auto &state = get_state();
    if (state.shut_down.load()) {
        return;
    }
    state.cache_limit.store(nbytes);
    if (state.cached.load() > nbytes) {
        empty_cache();
    }
}

std::size_t MemoryPoolRegistry::used_bytes() noexcept
{
    return get_state().used.load();
}

std::size_t MemoryPoolRegistry::cached_bytes() noexcept
{
    return get_state().cached.load();
}

bool MemoryPoolRegistry::try_cache(std::size_t nbytes) noexcept
{
    auto &state = get_state();
    std::size_t cur = state.cached.load();
    do {
        const std::size_t limit = state.cache_limit.load();
        if (cur > limit || nbytes > limit - cur) {
            return false;
        }
    } while (!state.cached.compare_exchange_weak(cur, cur + nbytes));
    return true;
}

void MemoryPoolRegistry::uncache(std::size_t nbytes) noexcept
{
    get_state().cached.fetch_sub(nbytes);
}

void MemoryPoolRegistry::add_used(std::size_t nbytes) noexcept
{
    get_state().used.fetch_add(nbytes);
}

void MemoryPoolRegistry::remove_used(std::size_t nbytes) noexcept
{
    get_state().used.fetch_sub(nbytes);
}

} // namespace memory
} // namespace dpctl
