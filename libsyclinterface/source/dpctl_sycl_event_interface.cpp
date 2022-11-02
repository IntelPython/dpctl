//===----- dpctl_sycl_event_interface.cpp - Implements C API for sycl::event =//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2022 Intel Corporation
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
#include "dpctl_error_handlers.h"
#include "dpctl_sycl_type_casters.hpp"
#include "dpctl_utils_helper.h"
#include <CL/sycl.hpp> /* SYCL headers   */
#include <vector>

using namespace sycl;

#undef EL
#define EL Event
#include "dpctl_vector_templ.cpp"
#undef EL

__dpctl_give DPCTLSyclEventRef DPCTLEvent_Create()
{
    DPCTLSyclEventRef ERef = nullptr;
    try {
        auto E = new event();
        ERef = wrap(E);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
    }
    return ERef;
}

void DPCTLEvent_Wait(__dpctl_keep DPCTLSyclEventRef ERef)
{
    if (ERef) {
        auto SyclEvent = unwrap(ERef);
        try {
            if (SyclEvent)
                SyclEvent->wait();
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    else {
        error_handler("Cannot wait for the event. DPCTLSyclEventRef as "
                      "input is a nullptr.",
                      __FILE__, __func__, __LINE__);
    }
}

void DPCTLEvent_WaitAndThrow(__dpctl_keep DPCTLSyclEventRef ERef)
{
    if (ERef) {
        auto SyclEvent = unwrap(ERef);
        try {
            if (SyclEvent)
                SyclEvent->wait_and_throw();
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    else {
        error_handler("Cannot wait_and_throw for the event. DPCTLSyclEventRef "
                      "as input is a nullptr.",
                      __FILE__, __func__, __LINE__);
    }
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
        error_handler("Cannot copy DPCTLSyclEventRef as input is a nullptr.",
                      __FILE__, __func__, __LINE__);
        return nullptr;
    }
    try {
        auto CopiedSyclEvent = new event(*SyclEvent);
        return wrap(CopiedSyclEvent);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
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
        error_handler("Backend cannot be looked up for a NULL event.", __FILE__,
                      __func__, __LINE__);
    }
    return BTy;
}

DPCTLSyclEventStatusType
DPCTLEvent_GetCommandExecutionStatus(__dpctl_keep DPCTLSyclEventRef ERef)
{
    DPCTLSyclEventStatusType ESTy =
        DPCTLSyclEventStatusType::DPCTL_UNKNOWN_STATUS;
    auto E = unwrap(ERef);
    if (E) {
        try {
            auto SyclESTy =
                E->get_info<sycl::info::event::command_execution_status>();
            ESTy = DPCTL_SyclEventStatusToDPCTLEventStatusType(SyclESTy);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    return ESTy;
}

uint64_t DPCTLEvent_GetProfilingInfoSubmit(__dpctl_keep DPCTLSyclEventRef ERef)
{
    uint64_t profilingInfoSubmit = 0;
    auto E = unwrap(ERef);
    if (E) {
        try {
            E->wait();
            profilingInfoSubmit = E->get_profiling_info<
                sycl::info::event_profiling::command_submit>();
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    return profilingInfoSubmit;
}

uint64_t DPCTLEvent_GetProfilingInfoStart(__dpctl_keep DPCTLSyclEventRef ERef)
{
    uint64_t profilingInfoStart = 0;
    auto E = unwrap(ERef);
    if (E) {
        try {
            E->wait();
            profilingInfoStart = E->get_profiling_info<
                sycl::info::event_profiling::command_start>();
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    return profilingInfoStart;
}

uint64_t DPCTLEvent_GetProfilingInfoEnd(__dpctl_keep DPCTLSyclEventRef ERef)
{
    uint64_t profilingInfoEnd = 0;
    auto E = unwrap(ERef);
    if (E) {
        try {
            E->wait();
            profilingInfoEnd = E->get_profiling_info<
                sycl::info::event_profiling::command_end>();
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    return profilingInfoEnd;
}

__dpctl_give DPCTLEventVectorRef
DPCTLEvent_GetWaitList(__dpctl_keep DPCTLSyclEventRef ERef)
{
    auto E = unwrap(ERef);
    if (!E) {
        error_handler("Cannot get wait list as input is a nullptr.", __FILE__,
                      __func__, __LINE__);
        return nullptr;
    }
    std::vector<DPCTLSyclEventRef> *EventsVectorPtr = nullptr;
    try {
        EventsVectorPtr = new std::vector<DPCTLSyclEventRef>();
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
    try {
        auto Events = E->get_wait_list();
        EventsVectorPtr->reserve(Events.size());
        for (const auto &Ev : Events) {
            EventsVectorPtr->emplace_back(wrap(new event(Ev)));
        }
        return wrap(EventsVectorPtr);
    } catch (std::exception const &e) {
        delete EventsVectorPtr;
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}
