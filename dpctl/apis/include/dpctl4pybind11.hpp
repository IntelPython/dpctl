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

/* This type caster associates ``sycl::queue`` C++ class with
 * :class:`dpctl.SyclQueue` for the purposes of generation of
 * Python bindings by pybind11.
 */
template <> struct type_caster<sycl::queue>
{
public:
    PYBIND11_TYPE_CASTER(sycl::queue, _("dpctl.SyclQueue"));

    bool load(handle src, bool)
    {
        PyObject *source = src.ptr();
        if (PyObject_TypeCheck(source, &PySyclQueueType)) {
            DPCTLSyclQueueRef QRef = SyclQueue_GetQueueRef(
                reinterpret_cast<PySyclQueueObject *>(source));
            sycl::queue *q = reinterpret_cast<sycl::queue *>(QRef);
            value = *q;
            return true;
        }
        else if (source == Py_None) {
            value = sycl::queue{};
            return true;
        }
        else {
            throw py::type_error(
                "Input is of unexpected type, expected dpctl.SyclQueue");
        }
    }

    static handle cast(sycl::queue src, return_value_policy, handle)
    {
        auto tmp = SyclQueue_Make(reinterpret_cast<DPCTLSyclQueueRef>(&src));
        return handle(reinterpret_cast<PyObject *>(tmp));
    }
};

/* This type caster associates ``sycl::device`` C++ class with
 * :class:`dpctl.SyclDevice` for the purposes of generation of
 * Python bindings by pybind11.
 */
template <> struct type_caster<sycl::device>
{
public:
    PYBIND11_TYPE_CASTER(sycl::device, _("dpctl.SyclDevice"));

    bool load(handle src, bool)
    {
        PyObject *source = src.ptr();
        if (PyObject_TypeCheck(source, &PySyclDeviceType)) {
            DPCTLSyclDeviceRef DRef = SyclDevice_GetDeviceRef(
                reinterpret_cast<PySyclDeviceObject *>(source));
            sycl::device *d = reinterpret_cast<sycl::device *>(DRef);
            value = *d;
            return true;
        }
        else if (source == Py_None) {
            value = sycl::device{};
            return true;
        }
        else {
            throw py::type_error(
                "Input is of unexpected type, expected dpctl.SyclDevice");
        }
    }

    static handle cast(sycl::device src, return_value_policy, handle)
    {
        auto tmp = SyclDevice_Make(reinterpret_cast<DPCTLSyclDeviceRef>(&src));
        return handle(reinterpret_cast<PyObject *>(tmp));
    }
};

/* This type caster associates ``sycl::context`` C++ class with
 * :class:`dpctl.SyclContext` for the purposes of generation of
 * Python bindings by pybind11.
 */
template <> struct type_caster<sycl::context>
{
public:
    PYBIND11_TYPE_CASTER(sycl::context, _("dpctl.SyclContext"));

    bool load(handle src, bool)
    {
        PyObject *source = src.ptr();
        if (PyObject_TypeCheck(source, &PySyclContextType)) {
            DPCTLSyclContextRef CRef = SyclContext_GetContextRef(
                reinterpret_cast<PySyclContextObject *>(source));
            sycl::context *ctx = reinterpret_cast<sycl::context *>(CRef);
            value = *ctx;
            return true;
        }
        else {
            throw py::type_error(
                "Input is of unexpected type, expected dpctl.SyclContext");
        }
    }

    static handle cast(sycl::context src, return_value_policy, handle)
    {
        auto tmp =
            SyclContext_Make(reinterpret_cast<DPCTLSyclContextRef>(&src));
        return handle(reinterpret_cast<PyObject *>(tmp));
    }
};

/* This type caster associates ``sycl::event`` C++ class with
 * :class:`dpctl.SyclEvent` for the purposes of generation of
 * Python bindings by pybind11.
 */
template <> struct type_caster<sycl::event>
{
public:
    PYBIND11_TYPE_CASTER(sycl::event, _("dpctl.SyclEvent"));

    bool load(handle src, bool)
    {
        PyObject *source = src.ptr();
        if (PyObject_TypeCheck(source, &PySyclEventType)) {
            DPCTLSyclEventRef ERef = SyclEvent_GetEventRef(
                reinterpret_cast<PySyclEventObject *>(source));
            sycl::event *ev = reinterpret_cast<sycl::event *>(ERef);
            value = *ev;
            return true;
        }
        else {
            throw py::type_error(
                "Input is of unexpected type, expected dpctl.SyclEvent");
        }
    }

    static handle cast(sycl::event src, return_value_policy, handle)
    {
        auto tmp = SyclEvent_Make(reinterpret_cast<DPCTLSyclEventRef>(&src));
        return handle(reinterpret_cast<PyObject *>(tmp));
    }
};
} // namespace detail
} // namespace pybind11
