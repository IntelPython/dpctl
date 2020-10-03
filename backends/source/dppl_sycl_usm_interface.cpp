//===--- dppl_sycl_usm_interface.cpp - DPPL-SYCL interface --*- C++ -*---===//
//
//               Python Data Parallel Processing Library (PyDPPL)
//
// Copyright 2020 Intel Corporation
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
/// dppl_sycl_usm_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dppl_sycl_usm_interface.h"
#include "Support/CBindingWrapping.h"

#include <CL/sycl.hpp>                /* SYCL headers   */

using namespace cl::sycl;

namespace
{
// Create wrappers for C Binding types (see CBindingWrapping.h).
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(queue, DPPLSyclQueueRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(context, DPPLSyclContextRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(void, DPPLSyclUSMRef)

} /* end of anonymous namespace */

__dppl_give DPPLSyclUSMRef
DPPLmalloc_shared (size_t size, __dppl_keep const DPPLSyclQueueRef QRef)
{
    auto Q = unwrap(QRef);
    auto Ptr = malloc_shared(size, *Q);
    return wrap(Ptr);
}

__dppl_give DPPLSyclUSMRef
DPPLmalloc_host (size_t size, __dppl_keep const DPPLSyclQueueRef QRef)
{
    auto Q = unwrap(QRef);
    auto Ptr = malloc_host(size, *Q);
    return wrap(Ptr);
}

__dppl_give DPPLSyclUSMRef
DPPLmalloc_device (size_t size, __dppl_keep const DPPLSyclQueueRef QRef)
{
    auto Q = unwrap(QRef);
    auto Ptr = malloc_device(size, *Q);
    return wrap(Ptr);
}

void DPPLfree_with_queue (__dppl_take DPPLSyclUSMRef MRef,
                          __dppl_keep const DPPLSyclQueueRef QRef)
{
    auto Ptr = unwrap(MRef);
    auto Q = unwrap(QRef);
    free(Ptr, *Q);
}

void DPPLfree_with_context (__dppl_take DPPLSyclUSMRef MRef,
                            __dppl_keep const DPPLSyclContextRef CRef)
{
    auto Ptr = unwrap(MRef);
    auto C = unwrap(CRef);
    free(Ptr, *C);
}

const char *
DPPLUSM_GetPointerType (__dppl_keep const DPPLSyclUSMRef MRef,
                        __dppl_keep const DPPLSyclContextRef CRef)
{
    auto Ptr = unwrap(MRef);
    auto C = unwrap(CRef);

    auto kind = get_pointer_type(Ptr, *C);
    switch(kind) {
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
