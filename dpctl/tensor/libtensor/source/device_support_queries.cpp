//===-- ------------ Implementation of _tensor_impl module  ----*-C++-*-/===//
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
//===--------------------------------------------------------------------===//
///
/// \file
/// This file defines functions of dpctl.tensor._tensor_impl extensions
//===--------------------------------------------------------------------===//

#include <string>

#include "dpctl4pybind11.hpp"
#include <CL/sycl.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

namespace
{

std::string _default_device_fp_type(sycl::device d)
{
    if (d.has(sycl::aspect::fp64)) {
        return "f8";
    }
    else {
        return "f4";
    }
}

std::string _default_device_int_type(sycl::device)
{
    return "i8";
}

std::string _default_device_complex_type(sycl::device d)
{
    if (d.has(sycl::aspect::fp64)) {
        return "c16";
    }
    else {
        return "c8";
    }
}

std::string _default_device_bool_type(sycl::device)
{
    return "b1";
}

sycl::device _extract_device(py::object arg)
{
    auto &api = dpctl::detail::dpctl_capi::get();

    PyObject *source = arg.ptr();
    if (api.PySyclQueue_Check_(source)) {
        sycl::queue q = py::cast<sycl::queue>(arg);
        return q.get_device();
    }
    else if (api.PySyclDevice_Check_(source)) {
        return py::cast<sycl::device>(arg);
    }
    else {
        throw py::type_error(
            "Expected type `dpctl.SyclQueue` or `dpctl.SyclDevice`.");
    }
}

} // namespace

std::string default_device_fp_type(py::object arg)
{
    sycl::device d = _extract_device(arg);
    return _default_device_fp_type(d);
}

std::string default_device_int_type(py::object arg)
{
    sycl::device d = _extract_device(arg);
    return _default_device_int_type(d);
}

std::string default_device_bool_type(py::object arg)
{
    sycl::device d = _extract_device(arg);
    return _default_device_bool_type(d);
}

std::string default_device_complex_type(py::object arg)
{
    sycl::device d = _extract_device(arg);
    return _default_device_complex_type(d);
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
