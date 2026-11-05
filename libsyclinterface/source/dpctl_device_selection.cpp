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
#include <charconv>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

namespace
{
static_assert(__SYCL_COMPILER_VERSION >= __SYCL_COMPILER_VERSION_REQUIRED,
              "The compiler does not meet minimum version requirement");

/*! Splits s on delim, dropping empty tokens. */
std::vector<std::string> tokenize(const std::string &s, char delim)
{
    std::vector<std::string> tokens;
    std::size_t start = 0;
    while (start <= s.size()) {
        auto end = s.find(delim, start);
        if (end == std::string::npos)
            end = s.size();
        if (end > start)
            tokens.emplace_back(s, start, end - start);
        start = end + 1;
    }
    return tokens;
}

std::optional<sycl::backend> backend_from_token(const std::string &tok)
{
    if (tok == "opencl")
        return sycl::backend::opencl;
    if (tok == "level_zero")
        return sycl::backend::ext_oneapi_level_zero;
    if (tok == "cuda")
        return sycl::backend::ext_oneapi_cuda;
    if (tok == "hip")
        return sycl::backend::ext_oneapi_hip;
    return std::nullopt;
}

std::optional<sycl::info::device_type>
device_type_from_token(const std::string &tok)
{
    if (tok == "cpu")
        return sycl::info::device_type::cpu;
    if (tok == "gpu")
        return sycl::info::device_type::gpu;
    if (tok == "accelerator")
        return sycl::info::device_type::accelerator;
    if (tok == "custom")
        return sycl::info::device_type::custom;
    return std::nullopt;
}

/*! Parses a non-negative decimal int; nullopt on any other input. */
std::optional<int> device_num_from_token(const std::string &tok)
{
    for (char c : tok) {
        if (c < '0' || c > '9')
            return std::nullopt;
    }
    int num = 0;
    const char *first = tok.data();
    const char *last = first + tok.size();
    auto [ptr, ec] = std::from_chars(first, last, num);
    if (ec != std::errc{} || ptr != last)
        return std::nullopt;
    return num;
}

bool matches_backend_and_type(const std::optional<sycl::backend> &be,
                              const std::optional<sycl::info::device_type> &dt,
                              const sycl::device &d)
{
    if (be && d.get_backend() != *be)
        return false;
    if (dt && d.get_info<sycl::info::device::device_type>() != *dt)
        return false;
    return true;
}

[[noreturn]] void throw_invalid_filter(const std::string &fs)
{
    throw std::invalid_argument(
        "Invalid filter string '" + fs +
        "'. Filters are comma-separated and have the form "
        "backend:device_type:device_num, where each component is optional.");
}
} // namespace

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

dpctl_filter_selector::dpctl_filter_selector(const std::string &fs)
{
    std::vector<std::pair<filter, std::optional<int>>> parsed;
    for (const auto &filter_str : tokenize(fs, ',')) {
        auto tokens = tokenize(filter_str, ':');
        if (tokens.size() > 3)
            throw_invalid_filter(fs);

        filter f;
        std::optional<int> device_num;
        for (const auto &tok : tokens) {
            if (auto be = backend_from_token(tok); be && !f.backend)
                f.backend = be;
            else if (auto dt = device_type_from_token(tok);
                     dt && !f.device_type)
                f.device_type = dt;
            else if (auto num = device_num_from_token(tok); num && !device_num)
                device_num = num;
            else
                throw_invalid_filter(fs);
        }
        parsed.emplace_back(std::move(f), device_num);
    }

    // Resolve device numbers; a filter whose device does not exist is dropped
    std::vector<sycl::device> devices;
    bool have_devices = false;
    for (auto &[f, device_num] : parsed) {
        if (device_num) {
            if (!have_devices) {
                devices = sycl::device::get_devices();
                have_devices = true;
            }
            int idx = 0;
            for (const auto &d : devices) {
                if (!matches_backend_and_type(f.backend, f.device_type, d))
                    continue;
                if (idx++ == *device_num) {
                    f.device = d;
                    break;
                }
            }
            if (!f.device)
                continue;
        }
        _filters.push_back(std::move(f));
    }
}

int dpctl_filter_selector::operator()(const sycl::device &d) const
{
    for (const auto &f : _filters) {
        if (matches_backend_and_type(f.backend, f.device_type, d) &&
            (!f.device || *f.device == d))
            return sycl::default_selector_v(d);
    }
    return REJECT_DEVICE;
}

} // namespace syclinterface
} // namespace dpctl
