//===-- dpctl_device_selectors.h - Device selector class declar. --*-C++-*- =//
//
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
/// This file declares classes for device selection.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Config/dpctl_config.h"
#include "Support/DllExport.h"
#include <CL/sycl.hpp>

namespace dpctl
{
namespace syclinterface
{

#if __SYCL_COMPILER_VERSION >= __SYCL_COMPILER_2023_SWITCHOVER

class DPCTL_API dpctl_device_selector
{
public:
    virtual ~dpctl_device_selector() = default;
    static constexpr int REJECT_DEVICE = -1;
    virtual int operator()(const sycl::device &) const;
};

class DPCTL_API dpctl_accelerator_selector : public dpctl_device_selector
{
public:
    dpctl_accelerator_selector() = default;
    int operator()(const sycl::device &d) const override;
};

class DPCTL_API dpctl_default_selector : public dpctl_device_selector
{
public:
    dpctl_default_selector() = default;
    int operator()(const sycl::device &d) const override;
};

class DPCTL_API dpctl_gpu_selector : public dpctl_device_selector
{
public:
    dpctl_gpu_selector() = default;
    int operator()(const sycl::device &d) const override;
};

class DPCTL_API dpctl_cpu_selector : public dpctl_device_selector
{
public:
    dpctl_cpu_selector() = default;
    int operator()(const sycl::device &d) const override;
};

class DPCTL_API dpctl_filter_selector : public dpctl_device_selector
{
public:
    dpctl_filter_selector(const std::string &fs) : _impl(fs) {}
    int operator()(const sycl::device &d) const override;

private:
    sycl::ext::oneapi::filter_selector _impl;
};

class DPCTL_API dpctl_host_selector : public dpctl_device_selector
{
public:
    dpctl_host_selector() = default;
    int operator()(const sycl::device &) const override;
};

#else

class DPCTL_API dpctl_device_selector : public sycl::device_selector
{
public:
    virtual ~dpctl_device_selector() = default;

    virtual int operator()(const sycl::device &device) const = 0;
};

class DPCTL_API dpctl_accelerator_selector : public dpctl_device_selector
{
public:
    dpctl_accelerator_selector() : _impl(){};
    int operator()(const sycl::device &d) const override;

private:
    sycl::accelerator_selector _impl;
};

class DPCTL_API dpctl_default_selector : public dpctl_device_selector
{
public:
    dpctl_default_selector() : _impl(){};
    int operator()(const sycl::device &d) const override;

private:
    sycl::default_selector _impl;
};

class DPCTL_API dpctl_gpu_selector : public dpctl_device_selector
{
public:
    dpctl_gpu_selector() : _impl(){};
    int operator()(const sycl::device &d) const override;

private:
    sycl::gpu_selector _impl;
};

class DPCTL_API dpctl_cpu_selector : public dpctl_device_selector
{
public:
    dpctl_cpu_selector() : _impl(){};
    int operator()(const sycl::device &d) const override;

private:
    sycl::cpu_selector _impl;
};

class DPCTL_API dpctl_filter_selector : public dpctl_device_selector
{
public:
    dpctl_filter_selector(const std::string &fs) : _impl(fs) {}

    int operator()(const sycl::device &d) const override;

private:
    sycl::ext::oneapi::filter_selector _impl;
};

class DPCTL_API dpctl_host_selector : public dpctl_device_selector
{
public:
    dpctl_host_selector() : _impl(){};
    int operator()(const sycl::device &d) const override;

private:
    sycl::host_selector _impl;
};

#endif

} // namespace syclinterface
} // namespace dpctl
