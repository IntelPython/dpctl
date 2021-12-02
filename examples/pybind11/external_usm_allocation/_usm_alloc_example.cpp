//==- _usm_alloc_example.cpp - Example of Pybind11 extension exposing   --===//
//   native USM allocation to Python in such a way that dpctl.memory
//   can form views into it.
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
/// This file implements Pybind11-generated extension that creates Python type
/// backed-up by C++ class DMatrix, which creates a USM allocation associated
/// with a given dpctl.SyclQueue. The Python object of this type implements
/// __sycl_usm_array_interface__, allowing dpctl.memory.as_usm_memory to form
/// a view into this allocation, and modify it from Python.
///
/// The DMatrix type object also implements `.tolist()` method which copies
/// content of the object into list of lists of Python floats.
///
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>

#include "dpctl4pybind11.hpp"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

struct DMatrix
{
    using alloc_t = sycl::usm_allocator<double, sycl::usm::alloc::shared>;
    using vec_t = std::vector<double, alloc_t>;

    DMatrix(sycl::queue &q, size_t rows, size_t columns)
        : n_(rows), m_(columns), q_(q), alloc_(q), vec_(n_ * m_, alloc_)
    {
    }
    ~DMatrix(){};
    DMatrix(const DMatrix &) = default;
    DMatrix(DMatrix &&) = default;

    size_t get_n() const
    {
        return n_;
    }
    size_t get_m() const
    {
        return m_;
    }
    vec_t &get_vector()
    {
        return vec_;
    }
    sycl::queue get_queue() const
    {
        return q_;
    }

    double get_element(size_t i, size_t j)
    {
        return vec_.at(i * m_ + j);
    }

private:
    size_t n_;
    size_t m_;
    sycl::queue q_;
    alloc_t alloc_;
    vec_t vec_;
};

DMatrix create_matrix(sycl::queue &q, size_t n, size_t m)
{
    return DMatrix(q, n, m);
}

py::dict construct_sua_iface(DMatrix &m)
{
    // need "version", "data", "shape", "typestr", "syclobj"
    py::tuple shape = py::make_tuple(m.get_n(), m.get_m());
    py::list data_entry(2);
    data_entry[0] = reinterpret_cast<size_t>(m.get_vector().data());
    data_entry[1] = true;
    auto syclobj = py::capsule(
        reinterpret_cast<void *>(new sycl::queue(m.get_queue())),
        "SyclQueueRef", [](PyObject *cap) {
            if (cap) {
                auto name = PyCapsule_GetName(cap);
                std::string name_s(name);
                if (name_s == "SyclQueueRef" or name_s == "used_SyclQueueRef") {
                    void *p = PyCapsule_GetPointer(cap, name);
                    delete reinterpret_cast<sycl::queue *>(p);
                }
            }
        });
    py::dict iface;
    iface["data"] = data_entry;
    iface["shape"] = shape;
    iface["strides"] = py::none();
    iface["version"] = 1;
    iface["typestr"] = "|f8";
    iface["syclobj"] = syclobj;

    return iface;
}

py::list tolist(DMatrix &m)
{
    size_t rows_count = m.get_n();
    size_t cols_count = m.get_m();
    py::list rows(rows_count);
    for (size_t i = 0; i < rows_count; ++i) {
        py::list row_i(cols_count);
        for (size_t j = 0; j < cols_count; ++j) {
            row_i[j] = m.get_element(i, j);
        }
        rows[i] = row_i;
    }
    return rows;
}

PYBIND11_MODULE(external_usm_alloc, m)
{
    // Import the dpctl extensions
    import_dpctl();

    py::class_<DMatrix> dm(m, "DMatrix");
    dm.def(py::init(&create_matrix),
           "DMatrix(dpctl.SyclQueue, n_rows, n_cols)");
    dm.def_property("__sycl_usm_array_interface__", &construct_sua_iface,
                    nullptr);
    dm.def("tolist", &tolist, "Return matrix a Python list of lists");
}
