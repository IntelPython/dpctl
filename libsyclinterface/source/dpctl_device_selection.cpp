//===- dpctl_device_selection.cpp - Implementation of classes   -*-C++-*- ===//
// dpctl_device_selector, dpctl_default_selector, etc.
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2025 Intel Corporation
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
/// This file implements device-selection classes declared in
/// dpctl_device_selection.hpp
///
//===----------------------------------------------------------------------===//

#include "dpctl_device_selection.hpp"
#include "Config/dpctl_config.h"
#include <sycl/sycl.hpp>

namespace
{
static_assert(__SYCL_COMPILER_VERSION >= __SYCL_COMPILER_VERSION_REQUIRED,
              "The compiler does not meet minimum version requirement");
}

namespace dpctl
{
namespace syclinterface
{

int dpctl_device_selector::operator()(const sycl::device &) const
{
    return REJECT_DEVICE;
}

int dpctl_accelerator_selector::operator()(const sycl::device &d) const
{
    return sycl::accelerator_selector_v(d);
}

int dpctl_default_selector::operator()(const sycl::device &d) const
{
    auto score = sycl::default_selector_v(d);
    return score;
}

int dpctl_gpu_selector::operator()(const sycl::device &d) const
{
    return sycl::gpu_selector_v(d);
}

int dpctl_cpu_selector::operator()(const sycl::device &d) const
{
    return sycl::cpu_selector_v(d);
}

int dpctl_filter_selector::operator()(const sycl::device &d) const
{
    return _impl(d);
}

} // namespace syclinterface
} // namespace dpctl
