//===--- _opaque_smart_ptr.hpp                                     --------===//
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
//===---------------------------------------------------------------------===//
///
/// \file
/// This file implements working with shared_ptr<void> with USM deleted
/// disguided as an opaque pointer.
///
//===----------------------------------------------------------------------===//

#pragma once

#ifndef __cplusplus
#error "C++ is required to compile this file"
#endif

#include "syclinterface/dpctl_sycl_type_casters.hpp"
#include "syclinterface/dpctl_sycl_types.h"
#include <memory>
#include <sycl/sycl.hpp>

namespace
{

class USMDeleter
{
public:
    USMDeleter() = delete;
    USMDeleter(const USMDeleter &) = default;
    USMDeleter(USMDeleter &&) = default;
    USMDeleter(const ::sycl::queue &queue) : _context(queue.get_context()) {}
    USMDeleter(const ::sycl::context &context) : _context(context) {}
    template <typename T> void operator()(T *ptr) const
    {
        ::sycl::free(ptr, _context);
    }

private:
    ::sycl::context _context;
};

} // end of anonymous namespace

void *OpaqueSmartPtr_Make(void *usm_ptr, const sycl::queue &q)
{
    USMDeleter _deleter(q);
    auto sptr = new std::shared_ptr<void>(usm_ptr, std::move(_deleter));

    return reinterpret_cast<void *>(sptr);
}

void *OpaqueSmartPtr_Make(void *usm_ptr, DPCTLSyclQueueRef QRef)
{
    sycl::queue *q_ptr = dpctl::syclinterface::unwrap<sycl::queue>(QRef);

    // make a copy of queue
    sycl::queue q{*q_ptr};

    void *res = OpaqueSmartPtr_Make(usm_ptr, q);

    return res;
}

void OpaqueSmartPtr_Delete(void *opaque_ptr)
{
    auto sptr = reinterpret_cast<std::shared_ptr<void> *>(opaque_ptr);

    delete sptr;
}

void *OpaqueSmartPtr_Copy(void *opaque_ptr)
{
    auto sptr = reinterpret_cast<std::shared_ptr<void> *>(opaque_ptr);
    auto copied_sptr = new std::shared_ptr<void>(*sptr);

    return reinterpret_cast<void *>(copied_sptr);
}

long OpaqueSmartPtr_UseCount(void *opaque_ptr)
{
    auto sptr = reinterpret_cast<std::shared_ptr<void> *>(opaque_ptr);
    return sptr->use_count();
}

void *OpaqueSmartPtr_Get(void *opaque_ptr)
{
    auto sptr = reinterpret_cast<std::shared_ptr<void> *>(opaque_ptr);

    return sptr->get();
}
