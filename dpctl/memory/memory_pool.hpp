//===--- memory_pool.hpp                                     -------------===//
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

#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <sycl/sycl.hpp>
#include <vector>

namespace dpctl
{
namespace memory
{

class MemoryPool : public std::enable_shared_from_this<MemoryPool>
{
private:
    sycl::context ctx_;
    sycl::device dev_;
    sycl::usm::alloc type_;

    mutable std::mutex pool_mutex_;
    std::map<std::size_t, std::vector<void *>> global_free_blocks_;

    std::vector<void *> sycl_allocations_;

    static constexpr std::size_t alloc_threshold_ = 1024 * 1024; // 1 MiB
    static constexpr size_t slab_size_ = 2 * 1024 * 1024;        // 16 MiB

    void *allocate_from_sycl(std::size_t size);

public:
    MemoryPool(const sycl::context &ctx,
               const sycl::device &dev,
               sycl::usm::alloc type)
        : ctx_(ctx), dev_(dev), type_(type)
    {
    }

    ~MemoryPool();

    MemoryPool(const MemoryPool &) = delete;
    MemoryPool &operator=(const MemoryPool &) = delete;

    void *allocate(std::size_t size);
    void free(void *ptr, std::size_t size);
    void empty_cache();

    const sycl::context &get_context() const { return ctx_; }
    const sycl::device &get_device() const { return dev_; }
    sycl::usm::alloc get_type() const { return type_; }
};

class USMPoolDeleter
{
    std::shared_ptr<MemoryPool> pool_;
    std::size_t size_;

public:
    USMPoolDeleter(std::shared_ptr<MemoryPool> pool, std::size_t size)
        : pool_(std::move(pool)), size_(size)
    {
    }

    void operator()(void *ptr) const
    {
        if (ptr) {
            pool_->free(ptr, size_);
        }
    }
};

} // namespace memory
} // namespace dpctl
