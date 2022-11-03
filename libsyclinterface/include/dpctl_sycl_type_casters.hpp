//===-- dpctl_sycl_type_casters.h - Defines casters between --------*-C++-*- =//
//  the opaque and the underlying types.
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
/// This file defines casters between opaque types and underlying SYCL types.
///
//===----------------------------------------------------------------------===//

#pragma once

#ifdef __cplusplus

#include "dpctl_sycl_types.h"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

namespace dpctl::syclinterface
{

#if __SYCL_COMPILER_VERSION >= 20221020

class dpctl_device_selector
{
public:
    virtual ~dpctl_device_selector() = default;
    static constexpr int REJECT_DEVICE = -1;
    virtual int operator()(const sycl::device &d) const
    {
        std::cout << "Outright rejecting "
                  << d.get_info<sycl::info::device::name>() << std::endl;
        return REJECT_DEVICE;
    };
};

class dpctl_accelerator_selector : public dpctl_device_selector
{
public:
    dpctl_accelerator_selector() = default;
    int operator()(const sycl::device &d) const override
    {
        return sycl::accelerator_selector_v(d);
    }
};

class dpctl_default_selector : public dpctl_device_selector
{
public:
    dpctl_default_selector() = default;
    int operator()(const sycl::device &d) const override
    {
        auto score = sycl::default_selector_v(d);
        std::cout << "Got score = " << score << std::endl;
        return score;
    }
};

class dpctl_gpu_selector : public dpctl_device_selector
{
public:
    dpctl_gpu_selector() = default;
    int operator()(const sycl::device &d) const override
    {
        return sycl::gpu_selector_v(d);
    }
};

class dpctl_cpu_selector : public dpctl_device_selector
{
public:
    dpctl_cpu_selector() = default;
    int operator()(const sycl::device &d) const override
    {
        return sycl::cpu_selector_v(d);
    }
};

class dpctl_filter_selector : public dpctl_device_selector
{
public:
    dpctl_filter_selector(const std::string &fs) : _impl(fs) {}

    int operator()(const sycl::device &d) const override
    {
        return _impl(d);
    }

private:
    sycl::ext::oneapi::filter_selector _impl;
};

class dpctl_host_selector : public dpctl_device_selector
{
public:
    dpctl_host_selector() = default;
    int operator()(const sycl::device &) const override
    {
        return REJECT_DEVICE;
    }
};

#else

class dpctl_device_selector : public sycl::device_selector
{
public:
    virtual ~dpctl_device_selector() = default;

    virtual int operator()(const sycl::device &device) const = 0;
};

class dpctl_accelerator_selector : public dpctl_device_selector
{
public:
    dpctl_accelerator_selector() : _impl(){};
    int operator()(const sycl::device &d) const
    {
        return _impl(d);
    }

private:
    sycl::accelerator_selector _impl;
};

class dpctl_default_selector : public dpctl_device_selector
{
public:
    dpctl_default_selector() : _impl(){};
    int operator()(const sycl::device &d) const
    {
        return _impl(d);
    }

private:
    sycl::default_selector _impl;
};

class dpctl_gpu_selector : public dpctl_device_selector
{
public:
    dpctl_gpu_selector() : _impl(){};
    int operator()(const sycl::device &d) const
    {
        return _impl(d);
    }

private:
    sycl::gpu_selector _impl;
};

class dpctl_cpu_selector : public dpctl_device_selector
{
public:
    dpctl_cpu_selector() : _impl(){};
    int operator()(const sycl::device &d) const
    {
        return _impl(d);
    }

private:
    sycl::cpu_selector _impl;
};

class dpctl_filter_selector : public dpctl_device_selector
{
public:
    dpctl_filter_selector(const std::string &fs) : _impl(fs) {}

    int operator()(const sycl::device &d) const
    {
        return _impl(d);
    }

private:
    sycl::ext::oneapi::filter_selector _impl;
};

class dpctl_host_selector : public dpctl_device_selector
{
public:
    dpctl_host_selector() : _impl(){};
    int operator()(const sycl::device &d) const
    {
        return _impl(d);
    }

private:
    sycl::host_selector _impl;
};

#endif

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
