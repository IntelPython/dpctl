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
#include <complex>
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

class usm_memory : public py::object
{
public:
    // Use macro once Pybind11 2.9.2 is released instead of code bewteen
    // START_TOKEN and END_TOKEN
    /*
       PYBIND11_OBJECT_CVT(
          usm_memory,
          py::object,
          [](PyObject *o) -> bool{ return PyObject_TypeCheck(o, &Py_MemoryType)
       != 0;},
          [](PyObject *o) -> PyObject* { return as_usm_memory(o); }
        )
     */
    // START_TOKEN

    // these constructors do not validate, but since borrowed_t and stolen_t are
    // protected struct members of the object, they can only be called
    // internally.
    usm_memory(py::handle h, borrowed_t) : py::object(h, borrowed_t{}) {}
    usm_memory(py::handle h, stolen_t) : py::object(h, stolen_t{}) {}

    static bool check_(py::handle h)
    {
        return h.ptr() != nullptr &&
               PyObject_TypeCheck(h.ptr(), &Py_MemoryType);
    }

    template <typename Policy_>
    /* NOLINTNEXTLINE(google-explicit-constructor) */
    usm_memory(const py::detail::accessor<Policy_> &a)
        : usm_memory(py::object(a))
    {
    }

    usm_memory(const py::object &o)
        : py::object(check_(o) ? o.inc_ref().ptr() : as_usm_memory(o.ptr()),
                     stolen_t{})
    {
        if (!m_ptr)
            throw py::error_already_set();
    }

    /* NOLINTNEXTLINE(google-explicit-constructor) */
    usm_memory(py::object &&o)
        : py::object(check_(o) ? o.release().ptr() : as_usm_memory(o.ptr()),
                     stolen_t{})
    {
        if (!m_ptr)
            throw py::error_already_set();
    }
    // END_TOKEN

    usm_memory() : py::object(default_constructed(), stolen_t{})
    {
        if (!m_ptr)
            throw py::error_already_set();
    }

    sycl::queue get_queue() const
    {
        Py_MemoryObject *mem_obj = reinterpret_cast<Py_MemoryObject *>(m_ptr);
        DPCTLSyclQueueRef QRef = Memory_GetQueueRef(mem_obj);
        sycl::queue *obj_q = reinterpret_cast<sycl::queue *>(QRef);
        return *obj_q;
    }

    char *get_pointer() const
    {
        Py_MemoryObject *mem_obj = reinterpret_cast<Py_MemoryObject *>(m_ptr);
        DPCTLSyclUSMRef MRef = Memory_GetUsmPointer(mem_obj);
        return reinterpret_cast<char *>(MRef);
    }

    size_t get_nbytes() const
    {
        Py_MemoryObject *mem_obj = reinterpret_cast<Py_MemoryObject *>(m_ptr);
        return Memory_GetNumBytes(mem_obj);
    }

protected:
    static PyObject *as_usm_memory(PyObject *o)
    {
        if (o == nullptr) {
            PyErr_SetString(PyExc_ValueError,
                            "cannot create a usm_memory from a nullptr");
            return nullptr;
        }
        py::module_ m = py::module_::import("dpctl.memory");
        auto convertor = m.attr("as_usm_memory");

        py::object res;
        try {
            res = convertor(py::handle(o));
        } catch (const py::error_already_set &e) {
            return nullptr;
        }
        return res.ptr();
    }

    static PyObject *default_constructed()
    {
        py::module_ m = py::module_::import("dpctl.memory");
        auto kl = m.attr("MemoryUSMDevice");
        py::object res;
        try {
            // allocate 1 byte
            res = kl(1);
        } catch (const py::error_already_set &e) {
            return nullptr;
        }
        return res.ptr();
    }
};

class usm_ndarray : public py::object
{
public:
    // In Pybind11 2.9.2 replace code between START_TOKEN and END_TOKEN with
    // macro
    /*
      PYBIND11_OBJECT(
          usm_ndarray,
          py::object,
          [](PyObject *o) -> bool {return PyObject_TypeCheck(o, &PyUSMArrayType)
      != 0;}
      )
     */

    // START_TOKEN
    static bool check_(py::handle h)
    {
        return h.ptr() != nullptr &&
               PyObject_TypeCheck(h.ptr(), &PyUSMArrayType);
    }

    // these constructors do not validate, but since borrowed_t and stolen_t are
    // protected struct members of the object, they can only be called
    // internally.
    usm_ndarray(py::handle h, borrowed_t) : py::object(h, borrowed_t{}) {}
    usm_ndarray(py::handle h, stolen_t) : py::object(h, stolen_t{}) {}

    template <typename Policy_>
    /* NOLINTNEXTLINE(google-explicit-constructor) */
    usm_ndarray(const py::detail::accessor<Policy_> &a)
        : usm_ndarray(py::object(a))
    {
    }

    usm_ndarray(const py::object &o) : py::object(o)
    {
        if (m_ptr && !check_(m_ptr))
            throw PYBIND11_OBJECT_CHECK_FAILED(usm_ndarray, m_ptr);
    }

    /* NOLINTNEXTLINE(google-explicit-constructor) */
    usm_ndarray(py::object &&o) : py::object(std::move(o))
    {
        if (m_ptr && !check_(m_ptr))
            throw PYBIND11_OBJECT_CHECK_FAILED(usm_ndarray, m_ptr);
    }
    // END_TOKEN

    usm_ndarray() : py::object(default_constructed(), stolen_t{})
    {
        if (!m_ptr)
            throw py::error_already_set();
    }

    char *get_data()
    {
        PyObject *raw_o = this->ptr();
        PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

        return UsmNDArray_GetData(raw_ar);
    }

    template <typename T> T *get_data()
    {
        return reinterpret_cast<T *>(get_data());
    }

    int get_ndim()
    {
        PyObject *raw_o = this->ptr();
        PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

        return UsmNDArray_GetNDim(raw_ar);
    }

    const py::ssize_t *get_shape_raw()
    {
        PyObject *raw_o = this->ptr();
        PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

        return UsmNDArray_GetShape(raw_ar);
    }

    py::ssize_t get_shape(int i)
    {
        auto shape_ptr = get_shape_raw();
        return shape_ptr[i];
    }

    const py::ssize_t *get_strides_raw()
    {
        PyObject *raw_o = this->ptr();
        PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

        return UsmNDArray_GetStrides(raw_ar);
    }

    std::pair<py::ssize_t, py::ssize_t> get_minmax_offsets()
    {
        PyObject *raw_o = this->ptr();
        PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

        int nd = UsmNDArray_GetNDim(raw_ar);
        const py::ssize_t *shape = UsmNDArray_GetShape(raw_ar);
        const py::ssize_t *strides = UsmNDArray_GetStrides(raw_ar);

        py::ssize_t offset_min = 0;
        py::ssize_t offset_max = 0;
        if (strides == nullptr) {
            py::ssize_t stride(1);
            for (int i = 0; i < nd; ++i) {
                offset_max += stride * (shape[i] - 1);
                stride *= shape[i];
            }
        }
        else {
            offset_min = UsmNDArray_GetOffset(raw_ar);
            offset_max = offset_min;
            for (int i = 0; i < nd; ++i) {
                py::ssize_t delta = strides[i] * (shape[i] - 1);
                if (strides[i] > 0) {
                    offset_max += delta;
                }
                else {
                    offset_min += delta;
                }
            }
        }
        return std::make_pair(offset_min, offset_max);
    }

    sycl::queue get_queue()
    {
        PyObject *raw_o = this->ptr();
        PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

        DPCTLSyclQueueRef QRef = UsmNDArray_GetQueueRef(raw_ar);
        return *(reinterpret_cast<sycl::queue *>(QRef));
    }

    int get_typenum()
    {
        PyObject *raw_o = this->ptr();
        PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

        return UsmNDArray_GetTypenum(raw_ar);
    }

    int get_flags()
    {
        PyObject *raw_o = this->ptr();
        PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

        return UsmNDArray_GetFlags(raw_ar);
    }

    int get_elemsize()
    {
        PyObject *raw_o = this->ptr();
        PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

        return UsmNDArray_GetElementSize(raw_ar);
    }

private:
    static PyObject *default_constructed()
    {
        py::module_ m = py::module_::import("dpctl.tensor");
        auto kl = m.attr("usm_ndarray");
        py::object res;
        try {
            // allocate 1 byte
            res = kl(py::make_tuple(), py::arg("dtype") = "u1");
        } catch (const py::error_already_set &e) {
            return nullptr;
        }
        return res.ptr();
    }
};
