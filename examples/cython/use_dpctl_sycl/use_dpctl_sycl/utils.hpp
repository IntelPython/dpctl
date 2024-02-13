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

#pragma once

#include <CL/sycl.hpp>
#include <string>

std::string get_device_name(sycl::device d)
{
    return d.get_info<sycl::info::device::name>();
}

std::string get_device_driver_version(sycl::device d)
{
    return d.get_info<sycl::info::device::driver_version>();
}

sycl::device *copy_device(const sycl::device &d)
{
    auto copy_ptr = new sycl::device(d);
    return copy_ptr;
}
