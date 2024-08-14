//===-- sycl_alloc_utils.cpp - Allocation utilities ---*-C++-*- ----------====//
//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2024 Intel Corporation
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
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines CIndexer_array, and CIndexer_vector classes, as well
/// iteration space simplifiers.
//===----------------------------------------------------------------------===//

#pragma once

#include <exception>
#include <iostream>

#include "sycl/sycl.hpp"

namespace dpctl
{
namespace tensor
{
namespace alloc_utils
{

template <typename T>
class usm_host_allocator : public sycl::usm_allocator<T, sycl::usm::alloc::host>
{
public:
    using baseT = sycl::usm_allocator<T, sycl::usm::alloc::host>;
    using baseT::baseT;

    template <typename U> struct rebind
    {
        typedef usm_host_allocator<U> other;
    };

    void deallocate(T *ptr, size_t n)
    {
        try {
            baseT::deallocate(ptr, n);
        } catch (const std::exception &e) {
            std::cerr
                << "Exception caught in `usm_host_allocator::deallocate`: "
                << e.what() << std::endl;
        }
    }
};

template <typename T>
void sycl_free_noexcept(T *ptr, const sycl::context &ctx) noexcept
{
    try {
        sycl::free(ptr, ctx);
    } catch (const std::exception &e) {
        std::cerr << "Call to sycl::free caught exception: " << e.what()
                  << std::endl;
    }
}

template <typename T> void sycl_free_noexcept(T *ptr, sycl::queue &q) noexcept
{
    sycl_free_noexcept(ptr, q.get_context());
}

} // end of namespace alloc_utils
} // end of namespace tensor
} // end of namespace dpctl
