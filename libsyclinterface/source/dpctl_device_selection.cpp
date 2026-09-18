//===- dpctl_device_selection.cpp - Implementation of classes   -*-C++-*- ===//
// dpctl_device_selector, dpctl_default_selector, etc.
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2022 Intel Corporation
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
#include <cstddef>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

namespace
{
#ifndef __ADAPTIVECPP__
static_assert(__SYCL_COMPILER_VERSION >= __SYCL_COMPILER_VERSION_REQUIRED,
              "The compiler does not meet minimum version requirement");
#endif
} // namespace

#ifdef __ADAPTIVECPP__
namespace
{

std::vector<std::string> split_string(const std::string &s, char sep)
{
    std::vector<std::string> terms{};
    std::string::size_type pos = 0;
    while (true) {
        auto next = s.find(sep, pos);
        if (next == std::string::npos) {
            terms.emplace_back(s.substr(pos));
            return terms;
        }
        terms.emplace_back(s.substr(pos, next - pos));
        pos = next + 1;
    }
}

bool parse_backend(const std::string &term, sycl::backend &backend)
{
    if (term == "opencl")
        backend = sycl::backend::ocl;
    else if (term == "level_zero")
        backend = sycl::backend::level_zero;
    else if (term == "cuda")
        backend = sycl::backend::cuda;
    else if (term == "hip")
        backend = sycl::backend::hip;
    else if (term == "openmp")
        backend = sycl::backend::omp;
    else
        return false;
    return true;
}

bool parse_device_type(const std::string &term,
                       sycl::info::device_type &device_type)
{
    if (term == "cpu")
        device_type = sycl::info::device_type::cpu;
    else if (term == "gpu")
        device_type = sycl::info::device_type::gpu;
    else if (term == "accelerator")
        device_type = sycl::info::device_type::accelerator;
    else if (term == "custom")
        device_type = sycl::info::device_type::custom;
    else
        return false;
    return true;
}

bool parse_index(const std::string &term, long long &index)
{
    if (term.empty() ||
        term.find_first_not_of("0123456789") != std::string::npos)
        return false;
    index = std::stoll(term);
    return true;
}

/*!
 * @brief Appends devices matched by a single "backend:device_type:index"
 * filter, each term of which is optional, to the selected vector.
 *
 * Devices are enumerated the way DPCTLDeviceMgr_GetDevices enumerates them, so
 * that an index term agrees with the relative id in a device's filter string.
 */
void resolve_filter(const std::string &filter,
                    std::vector<sycl::device> &selected)
{
    auto terms = split_string(filter, ':');
    sycl::backend backend{};
    sycl::info::device_type device_type{};
    long long index = -1;
    bool has_backend = false;
    bool has_device_type = false;

    std::size_t pos = 0;
    if (pos < terms.size() && parse_backend(terms[pos], backend)) {
        has_backend = true;
        ++pos;
    }
    if (pos < terms.size() && parse_device_type(terms[pos], device_type)) {
        has_device_type = true;
        ++pos;
    }
    if (pos < terms.size() && parse_index(terms[pos], index))
        ++pos;
    if (pos != terms.size())
        throw std::invalid_argument("Could not parse filter string '" + filter +
                                    "'");

    dpctl::syclinterface::dpctl_default_selector ranker;
    long long matched = 0;
    for (const auto &d : sycl::device::get_devices()) {
        if (ranker(d) < 0)
            continue;
        if (has_backend && d.get_platform().get_backend() != backend)
            continue;
        if (has_device_type &&
            d.get_info<sycl::info::device::device_type>() != device_type)
            continue;
        if (index < 0)
            selected.emplace_back(d);
        else if (matched++ == index) {
            selected.emplace_back(d);
            break;
        }
    }
}

} // namespace
#endif

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
#ifndef __ADAPTIVECPP__
    return sycl::accelerator_selector_v(d);
#else
    // AdaptiveCpp defines is_accelerator() as !is_cpu(), which makes its
    // accelerator_selector_v accept GPUs as well
    auto score = sycl::default_selector_v(d);
    if (score < 0 || d.get_info<sycl::info::device::device_type>() !=
                         sycl::info::device_type::accelerator)
        return REJECT_DEVICE;
    return score + 1;
#endif
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

#ifndef __ADAPTIVECPP__
dpctl_filter_selector::dpctl_filter_selector(const std::string &fs) : _impl(fs)
{
}
#else
dpctl_filter_selector::dpctl_filter_selector(const std::string &fs)
{
    for (const auto &filter : split_string(fs, ','))
        resolve_filter(filter, _matches);
}
#endif

int dpctl_filter_selector::operator()(const sycl::device &d) const
{
#ifndef __ADAPTIVECPP__
    return _impl(d);
#else
    for (const auto &m : _matches)
        if (m == d) {
            // scores of matched devices are shifted to be strictly positive,
            // since AdaptiveCpp scores the OpenMP host device 0
            auto score = sycl::default_selector_v(d);
            return (score < 0) ? REJECT_DEVICE : score + 1;
        }
    return REJECT_DEVICE;
#endif
}

} // namespace syclinterface
} // namespace dpctl
