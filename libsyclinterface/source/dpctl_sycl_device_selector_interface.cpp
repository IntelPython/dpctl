//=== dpctl_sycl_device_selector_interface.cpp -  device_selector C API    ===//
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
/// Implements constructors for SYCL's device selector classes.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_device_selector_interface.h"
#include "../helper/include/dpctl_error_handlers.h"
#include "Support/CBindingWrapping.h"
#include <CL/sycl.hpp> /* SYCL headers   */

using namespace cl::sycl;

namespace
{
// Create wrappers for C Binding types (see CBindingWrapping.h).
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(device_selector, DPCTLSyclDeviceSelectorRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(device, DPCTLSyclDeviceRef)

} /* end of anonymous namespace */

__dpctl_give DPCTLSyclDeviceSelectorRef DPCTLAcceleratorSelector_Create()
{
    try {
        auto Selector = new accelerator_selector();
        return wrap(Selector);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

__dpctl_give DPCTLSyclDeviceSelectorRef DPCTLDefaultSelector_Create()
{
    try {
        auto Selector = new default_selector();
        return wrap(Selector);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

__dpctl_give DPCTLSyclDeviceSelectorRef DPCTLCPUSelector_Create()
{
    try {
        auto Selector = new cpu_selector();
        return wrap(Selector);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

__dpctl_give DPCTLSyclDeviceSelectorRef
DPCTLFilterSelector_Create(__dpctl_keep const char *filter_str)
{
#if __SYCL_COMPILER_VERSION < 20210925
    using filter_selector_t = sycl::ONEAPI::filter_selector;
#else
    using filter_selector_t = sycl::ext::oneapi::filter_selector;
#endif
    try {
        auto Selector = new filter_selector_t(filter_str);
        return wrap(Selector);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

__dpctl_give DPCTLSyclDeviceSelectorRef DPCTLGPUSelector_Create()
{
    try {
        auto Selector = new gpu_selector();
        return wrap(Selector);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

__dpctl_give DPCTLSyclDeviceSelectorRef DPCTLHostSelector_Create()
{
    try {
        auto Selector = new host_selector();
        return wrap(Selector);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

int DPCTLDeviceSelector_Score(__dpctl_keep DPCTLSyclDeviceSelectorRef DSRef,
                              __dpctl_keep DPCTLSyclDeviceRef DRef)
{
    constexpr int REJECT_DEVICE_SCORE = -1;
    if (DSRef && DRef) {
        auto dev = *(unwrap(DRef));
        return (*unwrap(DSRef))(dev);
    }
    else
        return REJECT_DEVICE_SCORE;
}

void DPCTLDeviceSelector_Delete(__dpctl_take DPCTLSyclDeviceSelectorRef DSRef)
{
    auto Selector = unwrap(DSRef);
    delete Selector;
}
