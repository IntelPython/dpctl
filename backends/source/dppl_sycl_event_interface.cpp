//===--- dppl_sycl_event_interface.cpp - DPPL-SYCL interface --*- C++ -*---===//
//
//               Data Parallel Control Library (dpCtl)
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
/// dppl_sycl_event_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dppl_sycl_event_interface.h"
#include "Support/CBindingWrapping.h"

#include <CL/sycl.hpp>                /* SYCL headers   */

using namespace cl::sycl;

namespace
{
// Create wrappers for C Binding types (see CBindingWrapping.h)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(event, DPPLSyclEventRef)
} /* end of anonymous namespace */


void DPPLEvent_Wait (__dppl_keep DPPLSyclEventRef ERef)
{
    // \todo How to handle errors? E.g. when ERef is null or not a valid event.
    auto SyclEvent = unwrap(ERef);
    SyclEvent->wait();
}

void
DPPLEvent_Delete (__dppl_take DPPLSyclEventRef ERef)
{
    delete unwrap(ERef);
}
