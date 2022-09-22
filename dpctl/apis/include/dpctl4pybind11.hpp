//===----------- dpctl4pybind11.h - Headers for type pybind11 casters  -*-C-*-
//===//
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
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines imports for dcptl's Python C-API
//===----------------------------------------------------------------------===//

#pragma once

#include "dpctl_capi.h"
#include <CL/sycl.hpp>
#include <complex>
#include <memory>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

namespace pybind11
{
namespace detail
{

#define DPCTL_TYPE_CASTER(type, py_name)                                       \
protected:                                                                     \
    std::unique_ptr<type> value;                                               \
                                                                               \
public:                                                                        \
    static constexpr auto name = py_name;                                      \
    template <                                                                 \
        typename T_,                                                           \
        ::pybind11::detail::enable_if_t<                                       \
            std::is_same<type, ::pybind11::detail::remove_cv_t<T_>>::value,    \
            int> = 0>                                                          \
    static ::pybind11::handle cast(T_ *src,                                    \
                                   ::pybind11::return_value_policy policy,     \
                                   ::pybind11::handle parent)                  \
    {                                                                          \
        if (!src)                                                              \
            return ::pybind11::none().release();                               \
        if (policy == ::pybind11::return_value_policy::take_ownership) {       \
            auto h = cast(std::move(*src), policy, parent);                    \
            delete src;                                                        \
            return h;                                                          \
        }                                                                      \
        return cast(*src, policy, parent);                                     \
    }                                                                          \
    operator type *()                                                          \
    {                                                                          \
        return value.get();                                                    \
    } /* NOLINT(bugprone-macro-parentheses) */                                 \
    operator type &()                                                          \
    {                                                                          \
        return *value;                                                         \
    } /* NOLINT(bugprone-macro-parentheses) */                                 \
    operator type &&() &&                                                      \
    {                                                                          \
        return std::move(*value);                                              \
    } /* NOLINT(bugprone-macro-parentheses) */                                 \
    template <typename T_>                                                     \
    using cast_op_type = ::pybind11::detail::movable_cast_op_type<T_>

/* This type caster associates ``sycl::queue`` C++ class with
 * :class:`dpctl.SyclQueue` for the purposes of generation of
 * Python bindings by pybind11.
 */
template <> struct type_caster<sycl::queue>
{
public:
    bool load(handle src, bool)
    {
        PyObject *source = src.ptr();
        if (PyObject_TypeCheck(source, &PySyclQueueType)) {
            DPCTLSyclQueueRef QRef = SyclQueue_GetQueueRef(
                reinterpret_cast<PySyclQueueObject *>(source));
            value = std::make_unique<sycl::queue>(
                *(reinterpret_cast<sycl::queue *>(QRef)));
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

    DPCTL_TYPE_CASTER(sycl::queue, _("dpctl.SyclQueue"));
};

/* This type caster associates ``sycl::device`` C++ class with
 * :class:`dpctl.SyclDevice` for the purposes of generation of
 * Python bindings by pybind11.
 */
template <> struct type_caster<sycl::device>
{
public:
    bool load(handle src, bool)
    {
        PyObject *source = src.ptr();
        if (PyObject_TypeCheck(source, &PySyclDeviceType)) {
            DPCTLSyclDeviceRef DRef = SyclDevice_GetDeviceRef(
                reinterpret_cast<PySyclDeviceObject *>(source));
            value = std::make_unique<sycl::device>(
                *(reinterpret_cast<sycl::device *>(DRef)));
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

    DPCTL_TYPE_CASTER(sycl::device, _("dpctl.SyclDevice"));
};

/* This type caster associates ``sycl::context`` C++ class with
 * :class:`dpctl.SyclContext` for the purposes of generation of
 * Python bindings by pybind11.
 */
template <> struct type_caster<sycl::context>
{
public:
    bool load(handle src, bool)
    {
        PyObject *source = src.ptr();
        if (PyObject_TypeCheck(source, &PySyclContextType)) {
            DPCTLSyclContextRef CRef = SyclContext_GetContextRef(
                reinterpret_cast<PySyclContextObject *>(source));
            value = std::make_unique<sycl::context>(
                *(reinterpret_cast<sycl::context *>(CRef)));
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

    DPCTL_TYPE_CASTER(sycl::context, _("dpctl.SyclContext"));
};

/* This type caster associates ``sycl::event`` C++ class with
 * :class:`dpctl.SyclEvent` for the purposes of generation of
 * Python bindings by pybind11.
 */
template <> struct type_caster<sycl::event>
{
public:
    bool load(handle src, bool)
    {
        PyObject *source = src.ptr();
        if (PyObject_TypeCheck(source, &PySyclEventType)) {
            DPCTLSyclEventRef ERef = SyclEvent_GetEventRef(
                reinterpret_cast<PySyclEventObject *>(source));
            value = std::make_unique<sycl::event>(
                *(reinterpret_cast<sycl::event *>(ERef)));
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

    DPCTL_TYPE_CASTER(sycl::event, _("dpctl.SyclEvent"));
};
} // namespace detail
} // namespace pybind11

namespace dpctl
{

namespace detail
{

struct dpctl_api
{
public:
    static dpctl_api &get()
    {
        static dpctl_api api;
        return api;
    }

    py::object sycl_queue_()
    {
        return *sycl_queue;
    }
    py::object default_usm_memory_()
    {
        return *default_usm_memory;
    }
    py::object default_usm_ndarray_()
    {
        return *default_usm_ndarray;
    }
    py::object as_usm_memory_()
    {
        return *as_usm_memory;
    }

private:
    struct Deleter
    {
        void operator()(py::object *p) const
        {
            bool guard = (Py_IsInitialized() && !_Py_IsFinalizing());

            if (guard) {
                delete p;
            }
        }
    };

    std::shared_ptr<py::object> sycl_queue;
    std::shared_ptr<py::object> default_usm_memory;
    std::shared_ptr<py::object> default_usm_ndarray;
    std::shared_ptr<py::object> as_usm_memory;

    dpctl_api() : sycl_queue{}, default_usm_memory{}, default_usm_ndarray{}
    {
        import_dpctl();

        sycl::queue q_;
        py::object py_sycl_queue = py::cast(q_);
        sycl_queue = std::shared_ptr<py::object>(new py::object{py_sycl_queue},
                                                 Deleter{});

        py::module_ mod_memory = py::module_::import("dpctl.memory");
        py::object py_as_usm_memory = mod_memory.attr("as_usm_memory");
        as_usm_memory = std::shared_ptr<py::object>(
            new py::object{py_as_usm_memory}, Deleter{});

        auto mem_kl = mod_memory.attr("MemoryUSMHost");
        py::object py_default_usm_memory =
            mem_kl(1, py::arg("queue") = py_sycl_queue);
        default_usm_memory = std::shared_ptr<py::object>(
            new py::object{py_default_usm_memory}, Deleter{});

        py::module_ mod_usmarray =
            py::module_::import("dpctl.tensor._usmarray");
        auto tensor_kl = mod_usmarray.attr("usm_ndarray");

        py::object py_default_usm_ndarray =
            tensor_kl(py::tuple(), py::arg("dtype") = py::str("u1"),
                      py::arg("buffer") = py_default_usm_memory);

        default_usm_ndarray = std::shared_ptr<py::object>(
            new py::object{py_default_usm_ndarray}, Deleter{});
    }

public:
    dpctl_api(dpctl_api const &) = delete;
    void operator=(dpctl_api const &) = delete;
    ~dpctl_api(){};
};

} // namespace detail

namespace memory
{

// since PYBIND11_OBJECT_CVT uses error_already_set without namespace,
// this allows to avoid compilation error
using pybind11::error_already_set;

class usm_memory : public py::object
{
public:
    PYBIND11_OBJECT_CVT(
        usm_memory,
        py::object,
        [](PyObject *o) -> bool {
            return PyObject_TypeCheck(o, &Py_MemoryType) != 0;
        },
        [](PyObject *o) -> PyObject * { return as_usm_memory(o); })

    usm_memory()
        : py::object(::dpctl::detail::dpctl_api::get().default_usm_memory_(),
                     borrowed_t{})
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

        auto convertor = ::dpctl::detail::dpctl_api::get().as_usm_memory_();

        py::object res;
        try {
            res = convertor(py::handle(o));
        } catch (const py::error_already_set &e) {
            return nullptr;
        }
        return res.ptr();
    }
};

} // end namespace memory

namespace tensor
{

std::vector<py::ssize_t> c_contiguous_strides(int nd,
                                              const py::ssize_t *shape,
                                              py::ssize_t element_size = 1)
{
    if (nd > 0) {
        std::vector<py::ssize_t> c_strides(nd, element_size);
        for (int ic = nd - 1; ic > 0;) {
            py::ssize_t next_v = c_strides[ic] * shape[ic];
            c_strides[--ic] = next_v;
        }
        return c_strides;
    }
    else {
        return std::vector<py::ssize_t>();
    }
}

std::vector<py::ssize_t> f_contiguous_strides(int nd,
                                              const py::ssize_t *shape,
                                              py::ssize_t element_size = 1)
{
    if (nd > 0) {
        std::vector<py::ssize_t> f_strides(nd, element_size);
        for (int i = 0; i < nd - 1;) {
            py::ssize_t next_v = f_strides[i] * shape[i];
            f_strides[++i] = next_v;
        }
        return f_strides;
    }
    else {
        return std::vector<py::ssize_t>();
    }
}

std::vector<py::ssize_t>
c_contiguous_strides(const std::vector<py::ssize_t> &shape,
                     py::ssize_t element_size = 1)
{
    return c_contiguous_strides(shape.size(), shape.data(), element_size);
}

std::vector<py::ssize_t>
f_contiguous_strides(const std::vector<py::ssize_t> &shape,
                     py::ssize_t element_size = 1)
{
    return f_contiguous_strides(shape.size(), shape.data(), element_size);
}

class usm_ndarray : public py::object
{
public:
    PYBIND11_OBJECT(usm_ndarray, py::object, [](PyObject *o) -> bool {
        return PyObject_TypeCheck(o, &PyUSMArrayType) != 0;
    })

    usm_ndarray()
        : py::object(::dpctl::detail::dpctl_api::get().default_usm_ndarray_(),
                     borrowed_t{})
    {
        if (!m_ptr)
            throw py::error_already_set();
    }

    char *get_data() const
    {
        PyObject *raw_o = this->ptr();
        PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

        return UsmNDArray_GetData(raw_ar);
    }

    template <typename T> T *get_data() const
    {
        return reinterpret_cast<T *>(get_data());
    }

    int get_ndim() const
    {
        PyObject *raw_o = this->ptr();
        PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

        return UsmNDArray_GetNDim(raw_ar);
    }

    const py::ssize_t *get_shape_raw() const
    {
        PyObject *raw_o = this->ptr();
        PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

        return UsmNDArray_GetShape(raw_ar);
    }

    py::ssize_t get_shape(int i) const
    {
        auto shape_ptr = get_shape_raw();
        return shape_ptr[i];
    }

    const py::ssize_t *get_strides_raw() const
    {
        PyObject *raw_o = this->ptr();
        PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

        return UsmNDArray_GetStrides(raw_ar);
    }

    py::ssize_t get_size() const
    {
        PyObject *raw_o = this->ptr();
        PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

        int ndim = UsmNDArray_GetNDim(raw_ar);
        const py::ssize_t *shape = UsmNDArray_GetShape(raw_ar);

        py::ssize_t nelems = 1;
        for (int i = 0; i < ndim; ++i) {
            nelems *= shape[i];
        }

        assert(nelems >= 0);
        return nelems;
    }

    std::pair<py::ssize_t, py::ssize_t> get_minmax_offsets() const
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

    sycl::queue get_queue() const
    {
        PyObject *raw_o = this->ptr();
        PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

        DPCTLSyclQueueRef QRef = UsmNDArray_GetQueueRef(raw_ar);
        return *(reinterpret_cast<sycl::queue *>(QRef));
    }

    int get_typenum() const
    {
        PyObject *raw_o = this->ptr();
        PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

        return UsmNDArray_GetTypenum(raw_ar);
    }

    int get_flags() const
    {
        PyObject *raw_o = this->ptr();
        PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

        return UsmNDArray_GetFlags(raw_ar);
    }

    int get_elemsize() const
    {
        PyObject *raw_o = this->ptr();
        PyUSMArrayObject *raw_ar = reinterpret_cast<PyUSMArrayObject *>(raw_o);

        return UsmNDArray_GetElementSize(raw_ar);
    }
};

} // end namespace tensor

namespace utils
{

template <std::size_t num>
sycl::event keep_args_alive(sycl::queue q,
                            const py::object (&py_objs)[num],
                            const std::vector<sycl::event> &depends = {})
{
    sycl::event host_task_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        std::array<std::shared_ptr<py::handle>, num> shp_arr;
        for (std::size_t i = 0; i < num; ++i) {
            shp_arr[i] = std::make_shared<py::handle>(py_objs[i]);
            shp_arr[i]->inc_ref();
        }
        cgh.host_task([=]() {
            bool guard = (Py_IsInitialized() && !_Py_IsFinalizing());
            if (guard) {
                PyGILState_STATE gstate;
                gstate = PyGILState_Ensure();
                for (std::size_t i = 0; i < num; ++i) {
                    shp_arr[i]->dec_ref();
                }
                PyGILState_Release(gstate);
            }
        });
    });

    return host_task_ev;
}

template <std::size_t num>
bool queues_are_compatible(sycl::queue exec_q,
                           const sycl::queue (&alloc_qs)[num])
{
    for (std::size_t i = 0; i < num; ++i) {

        if (exec_q != alloc_qs[i]) {
            return false;
        }
    }
    return true;
}

} // end namespace utils

} // end namespace dpctl
