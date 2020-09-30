//===--- dppl_sycl_queue_interface.cpp - DPPL-SYCL interface --*- C++ -*---===//
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
/// dppl_sycl_queue_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dppl_sycl_queue_interface.h"
#include "Support/CBindingWrapping.h"

#include <CL/sycl.hpp>                /* SYCL headers   */

using namespace cl::sycl;

namespace
{
// Create wrappers for C Binding types (see CBindingWrapping.h).
 DEFINE_SIMPLE_CONVERSION_FUNCTIONS(queue, DPPLSyclQueueRef)

} /* end of anonymous namespace */

__dppl_give DPPLSyclDeviceRef
DPPLQueue_GetDevice (__dppl_keep const DPPLSyclQueueRef QRef)
{
    auto Q = unwrap(QRef);
    auto Device = new device(Q->get_device());
    return reinterpret_cast<DPPLSyclDeviceRef>(Device);
}

__dppl_give DPPLSyclContextRef
DPPLQueue_GetContext (__dppl_keep const DPPLSyclQueueRef QRef)
{
    auto Q = unwrap(QRef);
    auto Context = new context(Q->get_context());
    return reinterpret_cast<DPPLSyclContextRef>(Context);
}

/*!
 * Delete the passed in pointer after verifying it points to a sycl::queue.
 */
void DPPLQueue_Delete (__dppl_take DPPLSyclQueueRef QRef)
{
    delete unwrap(QRef);
}

void DPPLQueue_Memcpy (__dppl_take const DPPLSyclQueueRef QRef,
                       void *Dest, const void *Src, size_t Count)
{
    auto Q = unwrap(QRef);
    auto event = Q->memcpy(Dest, Src, Count);
    event.wait();
}
