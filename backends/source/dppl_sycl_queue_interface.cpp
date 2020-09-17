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
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(device, DPPLSyclDeviceRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(context, DPPLSyclContextRef)
} /* end of anonymous namespace */

__dppl_give DPPLSyclDeviceRef
DPPLGetDeviceFromQueue (__dppl_keep const DPPLSyclQueueRef QRef)
{
    auto Q = unwrap_queue(QRef);
    auto Device = new device(Q->get_device());
    return wrap_device(Device);
}

__dppl_give DPPLSyclContextRef
DPPLGetContextFromQueue (__dppl_keep const DPPLSyclQueueRef QRef)
{
    auto Q = unwrap_queue(QRef);
    auto Context = new context(Q->get_context());
    return wrap_context(Context);
}

/*!
 * Delete the passed in pointer after verifying it points to a sycl::queue.
 */
void DPPLDeleteSyclQueue (__dppl_take DPPLSyclQueueRef QRef)
{
    delete unwrap_queue(QRef);
}
