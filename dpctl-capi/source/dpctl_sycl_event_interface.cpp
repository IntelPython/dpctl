//===----- dpctl_sycl_event_interface.cpp - Implements C API for sycl::event =//
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
/// dpctl_sycl_event_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_event_interface.h"
#include "../helper/include/dpctl_utils_helper.h"
#include "Support/CBindingWrapping.h"
#include <CL/sycl.hpp> /* SYCL headers   */

using namespace cl::sycl;

namespace
{
// Create wrappers for C Binding types (see CBindingWrapping.h)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(event, DPCTLSyclEventRef)
} /* end of anonymous namespace */

__dpctl_give DPCTLSyclEventRef DPCTLEvent_Create()
{
    DPCTLSyclEventRef ERef = nullptr;
    try {
        auto E = new event();
        ERef = wrap(E);
    } catch (std::bad_alloc const &ba) {
        std::cerr << ba.what() << '\n';
    }
    return ERef;
}

void DPCTLEvent_Wait(__dpctl_keep DPCTLSyclEventRef ERef)
{
    // \todo How to handle errors? E.g. when ERef is null or not a valid event.
    auto SyclEvent = unwrap(ERef);
    SyclEvent->wait();
}

void DPCTLEvent_Delete(__dpctl_take DPCTLSyclEventRef ERef)
{
    delete unwrap(ERef);
}

__dpctl_give DPCTLSyclEventRef
DPCTLEvent_Copy(__dpctl_keep DPCTLSyclEventRef ERef)
{
    auto SyclEvent = unwrap(ERef);
    if (!SyclEvent) {
        std::cerr << "Cannot copy DPCTLSyclEventRef as input is a nullptr\n";
        return nullptr;
    }
    try {
        auto CopiedSyclEvent = new event(*SyclEvent);
        return wrap(CopiedSyclEvent);
    } catch (std::bad_alloc const &ba) {
        // \todo log error
        std::cerr << ba.what() << '\n';
        return nullptr;
    }
}

DPCTLSyclBackendType DPCTLEvent_GetBackend(__dpctl_keep DPCTLSyclEventRef ERef)
{
    DPCTLSyclBackendType BTy = DPCTLSyclBackendType::DPCTL_UNKNOWN_BACKEND;
    auto E = unwrap(ERef);
    if (E) {
        BTy = DPCTL_SyclBackendToDPCTLBackendType(E->get_backend());
    }
    else {
        std::cerr << "Backend cannot be looked up for a NULL event\n";
    }
    return BTy;
}
