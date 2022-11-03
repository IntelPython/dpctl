//=== dpctl_sycl_device_selector_interface.cpp -  device_selector C API    ===//
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
/// Implements constructors for SYCL's device selector classes.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_device_selector_interface.h"
#include "dpctl_device_selection.hpp"
#include "dpctl_error_handlers.h"
#include "dpctl_sycl_type_casters.hpp"
#include <CL/sycl.hpp> /* SYCL headers   */

using namespace sycl;

namespace
{
using namespace dpctl::syclinterface;
} // end of anonymous namespace

__dpctl_give DPCTLSyclDeviceSelectorRef DPCTLAcceleratorSelector_Create()
{
    try {
        auto Selector = new dpctl_accelerator_selector();
        return wrap<dpctl_device_selector>(Selector);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

__dpctl_give DPCTLSyclDeviceSelectorRef DPCTLDefaultSelector_Create()
{
    try {
        auto Selector = new dpctl_default_selector();
        return wrap<dpctl_device_selector>(Selector);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

__dpctl_give DPCTLSyclDeviceSelectorRef DPCTLCPUSelector_Create()
{
    try {
        auto Selector = new dpctl_cpu_selector();
        return wrap<dpctl_device_selector>(Selector);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

__dpctl_give DPCTLSyclDeviceSelectorRef
DPCTLFilterSelector_Create(__dpctl_keep const char *filter_str)
{
    using filter_selector_t = dpctl_filter_selector;
    try {
        auto Selector = new filter_selector_t(filter_str);
        return wrap<dpctl_device_selector>(Selector);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

__dpctl_give DPCTLSyclDeviceSelectorRef DPCTLGPUSelector_Create()
{
    try {
        auto Selector = new dpctl_gpu_selector();
        return wrap<dpctl_device_selector>(Selector);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

__dpctl_give DPCTLSyclDeviceSelectorRef DPCTLHostSelector_Create()
{
    try {
        auto Selector = new dpctl_host_selector();
        return wrap<dpctl_device_selector>(Selector);
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
        auto dev = *(unwrap<device>(DRef));
        return (*unwrap<dpctl_device_selector>(DSRef))(dev);
    }
    else
        return REJECT_DEVICE_SCORE;
}

void DPCTLDeviceSelector_Delete(__dpctl_take DPCTLSyclDeviceSelectorRef DSRef)
{
    auto Selector = unwrap<dpctl_device_selector>(DSRef);
    delete Selector;
}
