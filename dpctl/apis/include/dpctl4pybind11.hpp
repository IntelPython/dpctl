//===----------- dpctl4pybind11.h - Headers for type pybind11 casters  -*-C-*-
//===//
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
/// This file defines imports for dcptl's Python C-API
//===----------------------------------------------------------------------===//

#pragma once

#include "dpctl_capi.h"
#include <CL/sycl.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace pybind11
{
namespace detail
{
template <> struct type_caster<sycl::queue>
{
public:
    PYBIND11_TYPE_CASTER(sycl::queue, _("dpctl.SyclQueue"));

    bool load(handle src, bool)
    {
        PyObject *source = src.ptr();
        if (PyObject_TypeCheck(source, &PySyclQueueType)) {
            DPCTLSyclQueueRef QRef =
                get_queue_ref(reinterpret_cast<PySyclQueueObject *>(source));
            sycl::queue *q = reinterpret_cast<sycl::queue *>(QRef);
            value = *q;
            return true;
        }
        else {
            throw std::runtime_error(
                "Input is of unexpected type, expected egapi.Example");
        }
    }

    static handle cast(sycl::queue src, return_value_policy, handle)
    {
        auto tmp = make_SyclQueue(reinterpret_cast<DPCTLSyclQueueRef>(&src));
        return handle(reinterpret_cast<PyObject *>(tmp));
    }
};
} // namespace detail
} // namespace pybind11

namespace pybind11
{
namespace detail
{
template <> struct type_caster<sycl::device>
{
public:
    PYBIND11_TYPE_CASTER(sycl::device, _("dpctl.SyclDevice"));

    bool load(handle src, bool)
    {
        PyObject *source = src.ptr();
        if (PyObject_TypeCheck(source, &PySyclDeviceType)) {
            DPCTLSyclDeviceRef DRef =
                get_device_ref(reinterpret_cast<PySyclDeviceObject *>(source));
            sycl::device *d = reinterpret_cast<sycl::device *>(DRef);
            value = *d;
            return true;
        }
        else {
            throw std::runtime_error(
                "Input is of unexpected type, expected egapi.Example");
        }
    }

    static handle cast(sycl::device src, return_value_policy, handle)
    {
        auto tmp = make_SyclDevice(reinterpret_cast<DPCTLSyclDeviceRef>(&src));
        return handle(reinterpret_cast<PyObject *>(tmp));
    }
};
} // namespace detail
} // namespace pybind11
