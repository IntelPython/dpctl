//===--- memory_pool_registry.cpp --------===//
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
/// This file implements a registry for memory pools associated with a USM
/// type, SYCL device, and SYCL context.
///
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "memory_pool.hpp"
#include "memory_pool_registry.hpp"

namespace dpctl
{
namespace memory
{

MemoryPoolRegistry::RegistryState &MemoryPoolRegistry::get_state()
{
    static RegistryState state;
    return state;
}

std::shared_ptr<MemoryPool>
MemoryPoolRegistry::get_pool(const sycl::context &ctx,
                             const sycl::device &dev,
                             sycl::usm::alloc type)
{
    auto &state = get_state();
    PoolKey key{ctx, dev, type};

    std::lock_guard<std::mutex> lock(state.mtx);

    auto it = state.pools.find(key);
    if (it != state.pools.end()) {
        return it->second;
    }

    auto pool = std::make_shared<MemoryPool>(ctx, dev, type);
    state.pools[key] = pool;
    return pool;
}

void MemoryPoolRegistry::empty_cache()
{
    auto &state = get_state();

    std::lock_guard<std::mutex> lock(state.mtx);
    for (auto &[key, pool] : state.pools) {
        pool->empty_cache();
    }
}

} // namespace memory
} // namespace dpctl
