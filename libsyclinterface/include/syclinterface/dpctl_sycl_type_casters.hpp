//===-- dpctl_sycl_type_casters.h - Defines casters between --------*-C++-*- =//
//  the opaque and the underlying types.
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
/// This file defines casters between opaque types and underlying SYCL types.
///
//===----------------------------------------------------------------------===//

#pragma once

#ifdef __cplusplus

#include "dpctl_device_selection.hpp"
#include "dpctl_sycl_types.h"
#include <sycl/sycl.hpp>
#include <vector>

namespace dpctl::syclinterface
{

/*!
    @brief Creates two convenience templated functions to
    reinterpret_cast an opaque pointer to a pointer to a Sycl type
    and vice-versa.
*/
#define DEFINE_SIMPLE_CONVERSION_FUNCTIONS(ty, ref)                            \
    template <typename T,                                                      \
              std::enable_if_t<std::is_same<T, ty>::value, bool> = true>       \
    __attribute__((unused)) T *unwrap(ref P)                                   \
    {                                                                          \
        return reinterpret_cast<ty *>(P);                                      \
    }                                                                          \
    template <typename T,                                                      \
              std::enable_if_t<std::is_same<T, ty>::value, bool> = true>       \
    __attribute__((unused)) ref wrap(const ty *P)                              \
    {                                                                          \
        return reinterpret_cast<ref>(const_cast<ty *>(P));                     \
    }

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(dpctl_device_selector,
                                   DPCTLSyclDeviceSelectorRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(sycl::device, DPCTLSyclDeviceRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(sycl::context, DPCTLSyclContextRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(sycl::queue, DPCTLSyclQueueRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(void, DPCTLSyclUSMRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(sycl::platform, DPCTLSyclPlatformRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(sycl::event, DPCTLSyclEventRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(sycl::kernel, DPCTLSyclKernelRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(
    sycl::kernel_bundle<sycl::bundle_state::executable>,
    DPCTLSyclKernelBundleRef)

#include "dpctl_sycl_device_manager.h"
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(std::vector<DPCTLSyclDeviceRef>,
                                   DPCTLDeviceVectorRef)

#include "dpctl_sycl_platform_manager.h"
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(std::vector<DPCTLSyclPlatformRef>,
                                   DPCTLPlatformVectorRef)

#include "dpctl_sycl_event_interface.h"
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(std::vector<DPCTLSyclEventRef>,
                                   DPCTLEventVectorRef)

#endif

} // namespace dpctl::syclinterface
