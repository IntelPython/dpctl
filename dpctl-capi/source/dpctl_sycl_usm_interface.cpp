//===------ dpctl_sycl_usm_interface.cpp - Implements C API for USM ops    ===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2021 Intel Corporation
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
/// This file implements the data types and functions declared in
/// dpctl_sycl_usm_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_usm_interface.h"
#include "Support/CBindingWrapping.h"
#include "dpctl_sycl_device_interface.h"
#include <CL/sycl.hpp> /* SYCL headers   */

using namespace cl::sycl;

namespace
{
// Create wrappers for C Binding types (see CBindingWrapping.h).
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(queue, DPCTLSyclQueueRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(device, DPCTLSyclDeviceRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(context, DPCTLSyclContextRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(void, DPCTLSyclUSMRef)

} /* end of anonymous namespace */

__dpctl_give DPCTLSyclUSMRef
DPCTLmalloc_shared(size_t size, __dpctl_keep const DPCTLSyclQueueRef QRef)
{
    if (!QRef) {
        std::cerr << "Input QRef is nullptr\n";
        return nullptr;
    }
    try {
        auto Q = unwrap(QRef);
        auto Ptr = malloc_shared(size, *Q);
        return wrap(Ptr);
    } catch (feature_not_supported const &fns) {
        std::cerr << fns.what() << '\n';
        return nullptr;
    }
}

__dpctl_give DPCTLSyclUSMRef
DPCTLaligned_alloc_shared(size_t alignment,
                          size_t size,
                          __dpctl_keep const DPCTLSyclQueueRef QRef)
{
    if (!QRef) {
        std::cerr << "Input QRef is nullptr\n";
        return nullptr;
    }
    try {
        auto Q = unwrap(QRef);
        auto Ptr = aligned_alloc_shared(alignment, size, *Q);
        return wrap(Ptr);
    } catch (feature_not_supported const &fns) {
        std::cerr << fns.what() << '\n';
        return nullptr;
    }
}

__dpctl_give DPCTLSyclUSMRef
DPCTLmalloc_host(size_t size, __dpctl_keep const DPCTLSyclQueueRef QRef)
{
    if (!QRef) {
        std::cerr << "Input QRef is nullptr\n";
        return nullptr;
    }
    // SYCL 2020 spec: for devices without aspect::usm_host_allocations:
    // undefined behavior
    auto Q = unwrap(QRef);
    auto Ptr = malloc_host(size, *Q);
    return wrap(Ptr);
}

__dpctl_give DPCTLSyclUSMRef
DPCTLaligned_alloc_host(size_t alignment,
                        size_t size,
                        __dpctl_keep const DPCTLSyclQueueRef QRef)
{
    if (!QRef) {
        std::cerr << "Input QRef is nullptr\n";
        return nullptr;
    }
    // SYCL 2020 spec: for devices without aspect::usm_host_allocations:
    // undefined behavior
    auto Q = unwrap(QRef);
    auto Ptr = aligned_alloc_host(alignment, size, *Q);
    return wrap(Ptr);
}

__dpctl_give DPCTLSyclUSMRef
DPCTLmalloc_device(size_t size, __dpctl_keep const DPCTLSyclQueueRef QRef)
{
    if (!QRef) {
        std::cerr << "Input QRef is nullptr\n";
        return nullptr;
    }
    try {
        auto Q = unwrap(QRef);
        auto Ptr = malloc_device(size, *Q);
        return wrap(Ptr);
    } catch (feature_not_supported const &fns) {
        std::cerr << fns.what() << '\n';
        return nullptr;
    }
}

__dpctl_give DPCTLSyclUSMRef
DPCTLaligned_alloc_device(size_t alignment,
                          size_t size,
                          __dpctl_keep const DPCTLSyclQueueRef QRef)
{
    if (!QRef) {
        std::cerr << "Input QRef is nullptr\n";
        return nullptr;
    }
    try {
        auto Q = unwrap(QRef);
        auto Ptr = aligned_alloc_device(alignment, size, *Q);
        return wrap(Ptr);
    } catch (feature_not_supported const &fns) {
        std::cerr << fns.what() << '\n';
        return nullptr;
    }
}

void DPCTLfree_with_queue(__dpctl_take DPCTLSyclUSMRef MRef,
                          __dpctl_keep const DPCTLSyclQueueRef QRef)
{
    if (!QRef) {
        std::cerr << "Input QRef is nullptr\n";
        return;
    }
    if (!MRef) {
        std::cerr << "Input MRef is nullptr, nothing to free\n";
        return;
    }
    auto Ptr = unwrap(MRef);
    auto Q = unwrap(QRef);
    free(Ptr, *Q);
}

void DPCTLfree_with_context(__dpctl_take DPCTLSyclUSMRef MRef,
                            __dpctl_keep const DPCTLSyclContextRef CRef)
{
    if (!CRef) {
        std::cerr << "Input CRef is nullptr\n";
        return;
    }
    if (!MRef) {
        std::cerr << "Input MRef is nullptr, nothing to free\n";
        return;
    }
    auto Ptr = unwrap(MRef);
    auto C = unwrap(CRef);
    free(Ptr, *C);
}

const char *DPCTLUSM_GetPointerType(__dpctl_keep const DPCTLSyclUSMRef MRef,
                                    __dpctl_keep const DPCTLSyclContextRef CRef)
{
    if (!CRef) {
        std::cerr << "Input CRef is nullptr\n";
        return "unknown";
    }
    if (!MRef) {
        std::cerr << "Input MRef is nullptr\n";
        return "unknown";
    }
    auto Ptr = unwrap(MRef);
    auto C = unwrap(CRef);

    auto kind = get_pointer_type(Ptr, *C);
    switch (kind) {
    case usm::alloc::host:
        return "host";
    case usm::alloc::device:
        return "device";
    case usm::alloc::shared:
        return "shared";
    default:
        return "unknown";
    }
}

DPCTLSyclDeviceRef
DPCTLUSM_GetPointerDevice(__dpctl_keep const DPCTLSyclUSMRef MRef,
                          __dpctl_keep const DPCTLSyclContextRef CRef)
{
    if (!CRef) {
        std::cerr << "Input CRef is nullptr\n";
        return nullptr;
    }
    if (!MRef) {
        std::cerr << "Input MRef is nullptr\n";
        return nullptr;
    }

    auto Ptr = unwrap(MRef);
    auto C = unwrap(CRef);

    auto Dev = get_pointer_device(Ptr, *C);

    return wrap(new device(Dev));
}
