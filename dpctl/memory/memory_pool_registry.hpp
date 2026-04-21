//===--- memory_pool_registry.hpp                                 --------===//
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

#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "memory_pool.hpp"

namespace dpctl
{
namespace memory
{

struct PoolKey
{
    sycl::context ctx;
    sycl::device dev;
    sycl::usm::alloc type;

    bool operator==(const PoolKey &other) const
    {
        return type == other.type && dev == other.dev && ctx == other.ctx;
    }
};

struct PoolKeyHash
{
    std::size_t operator()(const PoolKey &key) const
    {
        std::size_t h1 = std::hash<int>()(static_cast<int>(key.type));
        std::size_t h2 = std::hash<sycl::device>()(key.dev);
        std::size_t h3 = std::hash<sycl::context>()(key.ctx);

        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

class MemoryPoolRegistry
{
private:
    struct RegistryState
    {
        std::mutex mtx;
        std::unordered_map<PoolKey, std::shared_ptr<MemoryPool>, PoolKeyHash>
            pools;
    };

    static RegistryState &get_state();

public:
    static std::shared_ptr<MemoryPool> get_pool(const sycl::context &ctx,
                                                const sycl::device &dev,
                                                sycl::usm::alloc type);

    static void empty_cache();
};

} // namespace memory
} // namespace dpctl
