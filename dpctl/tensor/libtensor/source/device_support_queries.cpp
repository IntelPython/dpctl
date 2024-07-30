//===-- ------------ Implementation of _tensor_impl module  ----*-C++-*-/===//
//
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
//
//===--------------------------------------------------------------------===//
///
/// \file
/// This file defines functions of dpctl.tensor._tensor_impl extensions
//===--------------------------------------------------------------------===//

#include <string>

#include "dpctl4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sycl/sycl.hpp>

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

namespace
{

std::string _default_device_fp_type(const sycl::device &d)
{
    if (d.has(sycl::aspect::fp64)) {
        return "f8";
    }
    else {
        return "f4";
    }
}

int get_numpy_major_version()
{
    namespace py = pybind11;

    py::module_ numpy = py::module_::import("numpy");
    py::str version_string = numpy.attr("__version__");
    py::module_ numpy_lib = py::module_::import("numpy.lib");

    py::object numpy_version = numpy_lib.attr("NumpyVersion")(version_string);
    int major_version = numpy_version.attr("major").cast<int>();

    return major_version;
}

std::string _default_device_int_type(const sycl::device &)
{
    const int np_ver = get_numpy_major_version();

    if (np_ver >= 2) {
        return "i8";
    }
    else {
        // code for numpy.dtype('long') to be consistent
        // with NumPy's default integer type across
        // platforms.
        return "l";
    }
}

std::string _default_device_uint_type(const sycl::device &)
{
    const int np_ver = get_numpy_major_version();

    if (np_ver >= 2) {
        return "u8";
    }
    else {
        // code for numpy.dtype('long') to be consistent
        // with NumPy's default integer type across
        // platforms.
        return "L";
    }
}

std::string _default_device_complex_type(const sycl::device &d)
{
    if (d.has(sycl::aspect::fp64)) {
        return "c16";
    }
    else {
        return "c8";
    }
}

std::string _default_device_bool_type(const sycl::device &)
{
    return "b1";
}

std::string _default_device_index_type(const sycl::device &)
{
    return "i8";
}

sycl::device _extract_device(const py::object &arg)
{
    auto const &api = dpctl::detail::dpctl_capi::get();

    PyObject *source = arg.ptr();
    if (api.PySyclQueue_Check_(source)) {
        const sycl::queue &q = py::cast<sycl::queue>(arg);
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

std::string default_device_fp_type(const py::object &arg)
{
    const sycl::device &d = _extract_device(arg);
    return _default_device_fp_type(d);
}

std::string default_device_int_type(const py::object &arg)
{
    const sycl::device &d = _extract_device(arg);
    return _default_device_int_type(d);
}

std::string default_device_uint_type(const py::object &arg)
{
    const sycl::device &d = _extract_device(arg);
    return _default_device_uint_type(d);
}

std::string default_device_bool_type(const py::object &arg)
{
    const sycl::device &d = _extract_device(arg);
    return _default_device_bool_type(d);
}

std::string default_device_complex_type(const py::object &arg)
{
    const sycl::device &d = _extract_device(arg);
    return _default_device_complex_type(d);
}

std::string default_device_index_type(const py::object &arg)
{
    const sycl::device &d = _extract_device(arg);
    return _default_device_index_type(d);
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
