//===----------- dpctl4pybind11.h - Headers for type pybind11 casters  -*-C-*-
//===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2021 Intel Corporation
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

#include <atomic>
#include <complex>
#include <cstddef> // for std::size_t for C++ linkage
#include <memory>
#include <mutex>
#include <pybind11/pybind11.h>
#include <stddef.h> // for size_t for C linkage
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

#if defined(SYCL_EXT_ONEAPI_BFLOAT16) && __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
#include <sycl/ext/oneapi/bfloat16.hpp>
// defined when the type caster for sycl::ext::oneapi::bfloat16 is available
#define DPCTL_HAS_BFLOAT16 1
#endif

namespace py = pybind11;

namespace dpctl
{
namespace detail
{

/*!
 * @brief Whether the interpreter can still be called into.
 *
 * A destructor that may run at shutdown, such as the one releasing
 * `dpctl_capi`'s objects, must check this before touching Python: asking for
 * the GIL while finalization runs may not return, and once finalization is over
 * the call is not safe at all.
 */
inline bool interpreter_is_live()
{
    const bool initialized = Py_IsInitialized();
#if PY_VERSION_HEX < 0x30d0000
    return initialized && !_Py_IsFinalizing();
#else
    return initialized && !Py_IsFinalizing();
#endif
}

class dpctl_capi
{
public:
    // dpctl type objects
    PyTypeObject *Py_SyclDeviceType_;
    PyTypeObject *PySyclDeviceType_;
    PyTypeObject *Py_SyclContextType_;
    PyTypeObject *PySyclContextType_;
    PyTypeObject *Py_SyclEventType_;
    PyTypeObject *PySyclEventType_;
    PyTypeObject *Py_SyclQueueType_;
    PyTypeObject *PySyclQueueType_;
    PyTypeObject *Py_MemoryType_;
    PyTypeObject *PyMemoryUSMDeviceType_;
    PyTypeObject *PyMemoryUSMSharedType_;
    PyTypeObject *PyMemoryUSMHostType_;
    PyTypeObject *PySyclKernelBundleType_;
    PyTypeObject *PySyclKernelType_;

    DPCTLSyclDeviceRef (*SyclDevice_GetDeviceRef_)(PySyclDeviceObject *);
    PySyclDeviceObject *(*SyclDevice_Make_)(DPCTLSyclDeviceRef);

    DPCTLSyclContextRef (*SyclContext_GetContextRef_)(PySyclContextObject *);
    PySyclContextObject *(*SyclContext_Make_)(DPCTLSyclContextRef);

    DPCTLSyclEventRef (*SyclEvent_GetEventRef_)(PySyclEventObject *);
    PySyclEventObject *(*SyclEvent_Make_)(DPCTLSyclEventRef);

    DPCTLSyclQueueRef (*SyclQueue_GetQueueRef_)(PySyclQueueObject *);
    PySyclQueueObject *(*SyclQueue_Make_)(DPCTLSyclQueueRef);
    int (*SyclQueue_DeferRelease_)(DPCTLSyclQueueRef,
                                   DPCTLSyclEventRef *,
                                   size_t,
                                   void (*)(void *),
                                   void *,
                                   DPCTLSyclEventRef);

    // memory
    DPCTLSyclUSMRef (*Memory_GetUsmPointer_)(Py_MemoryObject *);
    void *(*Memory_GetOpaquePointer_)(Py_MemoryObject *);
    DPCTLSyclContextRef (*Memory_GetContextRef_)(Py_MemoryObject *);
    DPCTLSyclQueueRef (*Memory_GetQueueRef_)(Py_MemoryObject *);
    size_t (*Memory_GetNumBytes_)(Py_MemoryObject *);
    PyObject *(*Memory_Make_)(DPCTLSyclUSMRef,
                              size_t,
                              DPCTLSyclQueueRef,
                              PyObject *);

    // compiler
    DPCTLSyclKernelRef (*SyclKernel_GetKernelRef_)(PySyclKernelObject *);
    PySyclKernelObject *(*SyclKernel_Make_)(DPCTLSyclKernelRef, const char *);

    DPCTLSyclKernelBundleRef (*SyclKernelBundle_GetKernelBundleRef_)(
        PySyclKernelBundleObject *);
    PySyclKernelBundleObject *(*SyclKernelBundle_Make_)(
        DPCTLSyclKernelBundleRef);

    bool PySyclDevice_Check_(PyObject *obj) const
    {
        return PyObject_TypeCheck(obj, PySyclDeviceType_) != 0;
    }
    bool PySyclContext_Check_(PyObject *obj) const
    {
        return PyObject_TypeCheck(obj, PySyclContextType_) != 0;
    }
    bool PySyclEvent_Check_(PyObject *obj) const
    {
        return PyObject_TypeCheck(obj, PySyclEventType_) != 0;
    }
    bool PySyclQueue_Check_(PyObject *obj) const
    {
        return PyObject_TypeCheck(obj, PySyclQueueType_) != 0;
    }
    bool PySyclKernel_Check_(PyObject *obj) const
    {
        return PyObject_TypeCheck(obj, PySyclKernelType_) != 0;
    }
    bool PySyclKernelBundle_Check_(PyObject *obj) const
    {
        return PyObject_TypeCheck(obj, PySyclKernelBundleType_) != 0;
    }

    ~dpctl_capi()
    {
        as_usm_memory_.reset();
        default_usm_memory_.reset();
        default_sycl_queue_.reset();
    };

    static auto &get()
    {
        static std::atomic<dpctl_capi *> global_capi{nullptr};
        dpctl_capi *capi_ptr = global_capi.load(std::memory_order_acquire);

        // fast path
        if (capi_ptr) {
            return *capi_ptr;
        }

        static std::mutex init_mtx;

        // initialization requires calls into Python C-API, so initializing
        // thread must hold the GIL, while other threads must not block on the
        // mutex while holding the GIL, as this creates a deadlock
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(init_mtx);

        // double check after acquiring lock
        capi_ptr = global_capi.load(std::memory_order_relaxed);

        if (!capi_ptr) {
            // acquire gil to safely call into Python C API
            py::gil_scoped_acquire acquire;

            // initialize C-API singleton
            // not freed, as it's kept until process termination
            capi_ptr = new dpctl_capi();

            global_capi.store(capi_ptr, std::memory_order_release);
        }

        return *capi_ptr;
    }

    py::object default_sycl_queue_pyobj() { return *default_sycl_queue_; }
    py::object default_usm_memory_pyobj() { return *default_usm_memory_; }
    py::object as_usm_memory_pyobj() { return *as_usm_memory_; }

private:
    struct Deleter
    {
        void operator()(py::object *p) const
        {
            if (interpreter_is_live()) {
                delete p;
            }
        }
    };

    std::shared_ptr<py::object> default_sycl_queue_;
    std::shared_ptr<py::object> default_usm_memory_;
    std::shared_ptr<py::object> as_usm_memory_;

    dpctl_capi()
        : Py_SyclDeviceType_(nullptr), PySyclDeviceType_(nullptr),
          Py_SyclContextType_(nullptr), PySyclContextType_(nullptr),
          Py_SyclEventType_(nullptr), PySyclEventType_(nullptr),
          Py_SyclQueueType_(nullptr), PySyclQueueType_(nullptr),
          Py_MemoryType_(nullptr), PyMemoryUSMDeviceType_(nullptr),
          PyMemoryUSMSharedType_(nullptr), PyMemoryUSMHostType_(nullptr),
          PySyclKernelBundleType_(nullptr), PySyclKernelType_(nullptr),
          SyclDevice_GetDeviceRef_(nullptr), SyclDevice_Make_(nullptr),
          SyclContext_GetContextRef_(nullptr), SyclContext_Make_(nullptr),
          SyclEvent_GetEventRef_(nullptr), SyclEvent_Make_(nullptr),
          SyclQueue_GetQueueRef_(nullptr), SyclQueue_Make_(nullptr),
          SyclQueue_DeferRelease_(nullptr), Memory_GetUsmPointer_(nullptr),
          Memory_GetOpaquePointer_(nullptr), Memory_GetContextRef_(nullptr),
          Memory_GetQueueRef_(nullptr), Memory_GetNumBytes_(nullptr),
          Memory_Make_(nullptr), SyclKernel_GetKernelRef_(nullptr),
          SyclKernel_Make_(nullptr),
          SyclKernelBundle_GetKernelBundleRef_(nullptr),
          SyclKernelBundle_Make_(nullptr), default_sycl_queue_{},
          default_usm_memory_{}, as_usm_memory_{}

    {
        // Import Cython-generated C-API for dpctl
        // This imports python modules and initializes
        // static variables such as function pointers for C-API,
        // e.g. SyclDevice_GetDeviceRef, etc.
        // and pointers to Python types, i.e. PySyclDeviceType, etc.
        if (import_dpctl() < 0) {
            throw py::error_already_set();
        }

        // Python type objects for classes implemented by dpctl
        this->Py_SyclDeviceType_ = &Py_SyclDeviceType;
        this->PySyclDeviceType_ = &PySyclDeviceType;
        this->Py_SyclContextType_ = &Py_SyclContextType;
        this->PySyclContextType_ = &PySyclContextType;
        this->Py_SyclEventType_ = &Py_SyclEventType;
        this->PySyclEventType_ = &PySyclEventType;
        this->Py_SyclQueueType_ = &Py_SyclQueueType;
        this->PySyclQueueType_ = &PySyclQueueType;
        this->Py_MemoryType_ = &Py_MemoryType;
        this->PyMemoryUSMDeviceType_ = &PyMemoryUSMDeviceType;
        this->PyMemoryUSMSharedType_ = &PyMemoryUSMSharedType;
        this->PyMemoryUSMHostType_ = &PyMemoryUSMHostType;
        this->PySyclKernelBundleType_ = &PySyclKernelBundleType;
        this->PySyclKernelType_ = &PySyclKernelType;

        // SyclDevice API
        this->SyclDevice_GetDeviceRef_ = SyclDevice_GetDeviceRef;
        this->SyclDevice_Make_ = SyclDevice_Make;

        // SyclContext API
        this->SyclContext_GetContextRef_ = SyclContext_GetContextRef;
        this->SyclContext_Make_ = SyclContext_Make;

        // SyclEvent API
        this->SyclEvent_GetEventRef_ = SyclEvent_GetEventRef;
        this->SyclEvent_Make_ = SyclEvent_Make;

        // SyclQueue API
        this->SyclQueue_GetQueueRef_ = SyclQueue_GetQueueRef;
        this->SyclQueue_Make_ = SyclQueue_Make;
        this->SyclQueue_DeferRelease_ = SyclQueue_DeferRelease;

        // dpctl.memory API
        this->Memory_GetUsmPointer_ = Memory_GetUsmPointer;
        this->Memory_GetOpaquePointer_ = Memory_GetOpaquePointer;
        this->Memory_GetContextRef_ = Memory_GetContextRef;
        this->Memory_GetQueueRef_ = Memory_GetQueueRef;
        this->Memory_GetNumBytes_ = Memory_GetNumBytes;
        this->Memory_Make_ = Memory_Make;

        // dpctl.compiler API
        this->SyclKernel_GetKernelRef_ = SyclKernel_GetKernelRef;
        this->SyclKernel_Make_ = SyclKernel_Make;
        this->SyclKernelBundle_GetKernelBundleRef_ =
            SyclKernelBundle_GetKernelBundleRef;
        this->SyclKernelBundle_Make_ = SyclKernelBundle_Make;

        // create shared pointers to python objects used in type-casters
        // for dpctl::memory::usm_memory
        sycl::queue q_{};
        PySyclQueueObject *py_q_tmp =
            SyclQueue_Make(reinterpret_cast<DPCTLSyclQueueRef>(&q_));
        const py::object &py_sycl_queue = py::reinterpret_steal<py::object>(
            reinterpret_cast<PyObject *>(py_q_tmp));

        default_sycl_queue_ = std::shared_ptr<py::object>(
            new py::object(py_sycl_queue), Deleter{});

        py::module_ mod_memory = py::module_::import("dpctl.memory");
        const py::object &py_as_usm_memory = mod_memory.attr("as_usm_memory");
        as_usm_memory_ = std::shared_ptr<py::object>(
            new py::object{py_as_usm_memory}, Deleter{});

        auto mem_kl = mod_memory.attr("MemoryUSMHost");
        const py::object &py_default_usm_memory =
            mem_kl(1, py::arg("queue") = py_sycl_queue);
        default_usm_memory_ = std::shared_ptr<py::object>(
            new py::object{py_default_usm_memory}, Deleter{});
    }

    dpctl_capi(dpctl_capi const &) = default;
    dpctl_capi &operator=(dpctl_capi const &) = default;
    dpctl_capi &operator=(dpctl_capi &&) = default;

}; // struct dpctl_capi

} // namespace detail
} // namespace dpctl

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
        auto const &api = ::dpctl::detail::dpctl_capi::get();
        if (api.PySyclQueue_Check_(source)) {
            DPCTLSyclQueueRef QRef = api.SyclQueue_GetQueueRef_(
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
        auto const &api = ::dpctl::detail::dpctl_capi::get();
        auto tmp =
            api.SyclQueue_Make_(reinterpret_cast<DPCTLSyclQueueRef>(&src));
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
        auto const &api = ::dpctl::detail::dpctl_capi::get();
        if (api.PySyclDevice_Check_(source)) {
            DPCTLSyclDeviceRef DRef = api.SyclDevice_GetDeviceRef_(
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
        auto const &api = ::dpctl::detail::dpctl_capi::get();
        auto tmp =
            api.SyclDevice_Make_(reinterpret_cast<DPCTLSyclDeviceRef>(&src));
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
        auto const &api = ::dpctl::detail::dpctl_capi::get();
        if (api.PySyclContext_Check_(source)) {
            DPCTLSyclContextRef CRef = api.SyclContext_GetContextRef_(
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
        auto const &api = ::dpctl::detail::dpctl_capi::get();
        auto tmp =
            api.SyclContext_Make_(reinterpret_cast<DPCTLSyclContextRef>(&src));
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
        auto const &api = ::dpctl::detail::dpctl_capi::get();
        if (api.PySyclEvent_Check_(source)) {
            DPCTLSyclEventRef ERef = api.SyclEvent_GetEventRef_(
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
        auto const &api = ::dpctl::detail::dpctl_capi::get();
        auto tmp =
            api.SyclEvent_Make_(reinterpret_cast<DPCTLSyclEventRef>(&src));
        return handle(reinterpret_cast<PyObject *>(tmp));
    }

    DPCTL_TYPE_CASTER(sycl::event, _("dpctl.SyclEvent"));
};

/* This type caster associates ``sycl::kernel`` C++ class with
 * :class:`dpctl.compiler.SyclKernel` for the purposes of generation of
 * Python bindings by pybind11.
 */
template <> struct type_caster<sycl::kernel>
{
public:
    bool load(handle src, bool)
    {
        PyObject *source = src.ptr();
        auto const &api = ::dpctl::detail::dpctl_capi::get();
        if (api.PySyclKernel_Check_(source)) {
            DPCTLSyclKernelRef KRef = api.SyclKernel_GetKernelRef_(
                reinterpret_cast<PySyclKernelObject *>(source));
            value = std::make_unique<sycl::kernel>(
                *(reinterpret_cast<sycl::kernel *>(KRef)));
            return true;
        }
        else {
            throw py::type_error("Input is of unexpected type, expected "
                                 "dpctl.compiler.SyclKernel");
        }
    }

    static handle cast(sycl::kernel src, return_value_policy, handle)
    {
        auto const &api = ::dpctl::detail::dpctl_capi::get();
        auto tmp =
            api.SyclKernel_Make_(reinterpret_cast<DPCTLSyclKernelRef>(&src),
                                 "dpctl4pybind11_kernel");
        return handle(reinterpret_cast<PyObject *>(tmp));
    }

    DPCTL_TYPE_CASTER(sycl::kernel, _("dpctl.compiler.SyclKernel"));
};

/* This type caster associates
 * ``sycl::kernel_bundle<sycl::bundle_state::executable>`` C++ class with
 * :class:`dpctl.compiler.SyclKernelBundle` for the purposes of generation of
 * Python bindings by pybind11.
 */
template <>
struct type_caster<sycl::kernel_bundle<sycl::bundle_state::executable>>
{
public:
    bool load(handle src, bool)
    {
        PyObject *source = src.ptr();
        auto const &api = ::dpctl::detail::dpctl_capi::get();
        if (api.PySyclKernelBundle_Check_(source)) {
            DPCTLSyclKernelBundleRef KBRef =
                api.SyclKernelBundle_GetKernelBundleRef_(
                    reinterpret_cast<PySyclKernelBundleObject *>(source));
            value = std::make_unique<
                sycl::kernel_bundle<sycl::bundle_state::executable>>(
                *(reinterpret_cast<
                    sycl::kernel_bundle<sycl::bundle_state::executable> *>(
                    KBRef)));
            return true;
        }
        else {
            throw py::type_error("Input is of unexpected type, expected "
                                 "dpctl.compiler.SyclKernelBundle");
        }
    }

    static handle cast(sycl::kernel_bundle<sycl::bundle_state::executable> src,
                       return_value_policy,
                       handle)
    {
        auto const &api = ::dpctl::detail::dpctl_capi::get();
        auto tmp = api.SyclKernelBundle_Make_(
            reinterpret_cast<DPCTLSyclKernelBundleRef>(&src));
        return handle(reinterpret_cast<PyObject *>(tmp));
    }

    DPCTL_TYPE_CASTER(sycl::kernel_bundle<sycl::bundle_state::executable>,
                      _("dpctl.compiler.SyclKernelBundle"));
};

/* This type caster associates
 * ``sycl::half`` C++ class with Python :class:`float` for the purposes
 * of generation of Python bindings by pybind11.
 */
template <> struct type_caster<sycl::half>
{
public:
    bool load(handle src, bool convert)
    {
        double py_value;

        if (!src) {
            return false;
        }

        PyObject *source = src.ptr();

        if (convert || PyFloat_Check(source)) {
            py_value = PyFloat_AsDouble(source);
        }
        else {
            return false;
        }

        bool py_err = (py_value == double(-1)) && PyErr_Occurred();

        if (py_err) {
            PyErr_Clear();
            if (convert && (PyNumber_Check(source) != 0)) {
                auto tmp = reinterpret_steal<object>(PyNumber_Float(source));
                return load(tmp, false);
            }
            return false;
        }
        value = static_cast<sycl::half>(py_value);
        return true;
    }

    static handle cast(sycl::half src, return_value_policy, handle)
    {
        return PyFloat_FromDouble(static_cast<double>(src));
    }

    PYBIND11_TYPE_CASTER(sycl::half, _("float"));
};

#ifdef DPCTL_HAS_BFLOAT16
/* This type caster associates the ``sycl::ext::oneapi::bfloat16`` C++ class
 * with Python :class:`float` for the purposes of generation of Python bindings
 * by pybind11. Any object convertible to :class:`float`, such as an
 * ``ml_dtypes.bfloat16`` scalar, is accepted.
 */
template <> struct type_caster<sycl::ext::oneapi::bfloat16>
{
public:
    bool load(handle src, bool convert)
    {
        using bfloat16_t = sycl::ext::oneapi::bfloat16;

        double py_value;

        if (!src) {
            return false;
        }

        PyObject *source = src.ptr();

        if (convert || PyFloat_Check(source)) {
            py_value = PyFloat_AsDouble(source);
        }
        else {
            return false;
        }

        bool py_err = (py_value == double(-1)) && PyErr_Occurred();

        if (py_err) {
            PyErr_Clear();
            if (convert && (PyNumber_Check(source) != 0)) {
                auto tmp = reinterpret_steal<object>(PyNumber_Float(source));
                return load(tmp, false);
            }
            return false;
        }
        value = bfloat16_t(static_cast<float>(py_value));
        return true;
    }

    static handle
    cast(sycl::ext::oneapi::bfloat16 src, return_value_policy, handle)
    {
        return PyFloat_FromDouble(static_cast<double>(static_cast<float>(src)));
    }

    PYBIND11_TYPE_CASTER(sycl::ext::oneapi::bfloat16, _("float"));
};
#endif

} // namespace detail
} // namespace pybind11

namespace dpctl
{

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
            return PyObject_TypeCheck(
                       o, ::dpctl::detail::dpctl_capi::get().Py_MemoryType_) !=
                   0;
        },
        [](PyObject *o) -> PyObject * { return as_usm_memory(o); })

    usm_memory()
        : py::object(
              ::dpctl::detail::dpctl_capi::get().default_usm_memory_pyobj(),
              borrowed_t{})
    {
        if (!m_ptr)
            throw py::error_already_set();
    }

    /*! @brief Create usm_memory object from shared pointer that manages
     *  lifetime of the USM allocation.
     */
    usm_memory(void *usm_ptr,
               std::size_t nbytes,
               const sycl::queue &q,
               std::shared_ptr<void> shptr)
    {
        auto const &api = ::dpctl::detail::dpctl_capi::get();
        DPCTLSyclUSMRef usm_ref = reinterpret_cast<DPCTLSyclUSMRef>(usm_ptr);
        auto q_uptr = std::make_unique<sycl::queue>(q);
        DPCTLSyclQueueRef QRef =
            reinterpret_cast<DPCTLSyclQueueRef>(q_uptr.get());

        auto vacuous_destructor = []() {};
        py::capsule mock_owner(vacuous_destructor);

        // create memory object owned by mock_owner, it is a new reference
        PyObject *_memory =
            api.Memory_Make_(usm_ref, nbytes, QRef, mock_owner.ptr());
        auto ref_count_decrementer = [](PyObject *o) noexcept { Py_DECREF(o); };

        using py_uptrT =
            std::unique_ptr<PyObject, decltype(ref_count_decrementer)>;

        if (!_memory) {
            throw py::error_already_set();
        }

        auto memory_uptr = py_uptrT(_memory, ref_count_decrementer);
        std::shared_ptr<void> *opaque_ptr = new std::shared_ptr<void>(shptr);

        Py_MemoryObject *memobj = reinterpret_cast<Py_MemoryObject *>(_memory);
        // replace mock_owner capsule as the owner
        memobj->refobj = Py_None;
        // set opaque ptr field, usm_memory now knowns that USM is managed
        // by smart pointer
        memobj->_opaque_ptr = reinterpret_cast<void *>(opaque_ptr);

        // _memory will delete created copies of sycl::queue, and
        // std::shared_ptr and the deleter of the shared_ptr<void> is
        // supposed to free the USM allocation
        m_ptr = _memory;
        q_uptr.release();
        memory_uptr.release();
    }

    sycl::queue get_queue() const
    {
        Py_MemoryObject *mem_obj = reinterpret_cast<Py_MemoryObject *>(m_ptr);
        auto const &api = ::dpctl::detail::dpctl_capi::get();
        DPCTLSyclQueueRef QRef = api.Memory_GetQueueRef_(mem_obj);
        sycl::queue *obj_q = reinterpret_cast<sycl::queue *>(QRef);
        return *obj_q;
    }

    char *get_pointer() const
    {
        Py_MemoryObject *mem_obj = reinterpret_cast<Py_MemoryObject *>(m_ptr);
        auto const &api = ::dpctl::detail::dpctl_capi::get();
        DPCTLSyclUSMRef MRef = api.Memory_GetUsmPointer_(mem_obj);
        return reinterpret_cast<char *>(MRef);
    }

    std::size_t get_nbytes() const
    {
        auto const &api = ::dpctl::detail::dpctl_capi::get();
        Py_MemoryObject *mem_obj = reinterpret_cast<Py_MemoryObject *>(m_ptr);
        return api.Memory_GetNumBytes_(mem_obj);
    }

    bool is_managed_by_smart_ptr() const
    {
        auto const &api = ::dpctl::detail::dpctl_capi::get();
        Py_MemoryObject *mem_obj = reinterpret_cast<Py_MemoryObject *>(m_ptr);
        const void *opaque_ptr = api.Memory_GetOpaquePointer_(mem_obj);

        return bool(opaque_ptr);
    }

    const std::shared_ptr<void> &get_smart_ptr_owner() const
    {
        auto const &api = ::dpctl::detail::dpctl_capi::get();
        Py_MemoryObject *mem_obj = reinterpret_cast<Py_MemoryObject *>(m_ptr);
        void *opaque_ptr = api.Memory_GetOpaquePointer_(mem_obj);

        if (opaque_ptr) {
            auto shptr_ptr =
                reinterpret_cast<std::shared_ptr<void> *>(opaque_ptr);
            return *shptr_ptr;
        }
        else {
            throw std::runtime_error(
                "Memory object does not have smart pointer "
                "managing lifetime of USM allocation");
        }
    }

protected:
    static PyObject *as_usm_memory(PyObject *o)
    {
        if (o == nullptr) {
            PyErr_SetString(PyExc_ValueError,
                            "cannot create a usm_memory from a nullptr");
            return nullptr;
        }

        auto converter =
            ::dpctl::detail::dpctl_capi::get().as_usm_memory_pyobj();

        py::object res;
        try {
            res = converter(py::handle(o));
        } catch (const py::error_already_set &e) {
            return nullptr;
        }
        return res.ptr();
    }
};

} // end namespace memory

namespace utils
{

namespace detail
{

struct ManagedMemory
{

    static bool is_usm_managed_by_shared_ptr(const py::object &h)
    {
        if (py::isinstance<dpctl::memory::usm_memory>(h)) {
            const auto &usm_memory_inst =
                py::cast<dpctl::memory::usm_memory>(h);
            return usm_memory_inst.is_managed_by_smart_ptr();
        }
        return false;
    }

    static const std::shared_ptr<void> &extract_shared_ptr(const py::object &h)
    {
        if (py::isinstance<dpctl::memory::usm_memory>(h)) {
            const auto &usm_memory_inst =
                py::cast<dpctl::memory::usm_memory>(h);
            return usm_memory_inst.get_smart_ptr_owner();
        }
        throw std::runtime_error(
            "Attempted extraction of shared_ptr on an unrecognized type");
    }
};

/*!
 * @brief What `keep_args_alive` holds for the work using its objects.
 */
template <std::size_t num> struct held_args
{
    std::size_t n_usm_owners = 0;
    std::array<std::shared_ptr<void>, num> usm_owners{};
    std::size_t n_objects = 0;
    std::array<py::handle, num> objects{};

    /*!
     * @brief Drop what `self` holds, and delete it.
     */
    static void release(void *self)
    {
        std::unique_ptr<held_args> held(static_cast<held_args *>(self));

        for (std::size_t i = 0; i < held->n_usm_owners; ++i) {
            held->usm_owners[i].reset();
        }
        for (std::size_t i = 0; i < held->n_objects; ++i) {
            held->objects[i].dec_ref();
        }
    }
};

} // end of namespace detail

/*!
 * @brief Call `release(data)` once the work covered by `depends` completes.
 *
 * Submits an empty kernel to `q` after `depends`, and returns its event, which
 * is what the release waits for. `release` is called with the GIL held.
 *
 * Throws if nothing was deferred, in which case `release` is never called.
 */
inline sycl::event defer_release(sycl::queue &q,
                                 const std::vector<sycl::event> &depends,
                                 void (*release)(void *),
                                 void *data)
{
    std::vector<DPCTLSyclEventRef> dep_refs;
    dep_refs.reserve(depends.size());
    for (const sycl::event &e : depends) {
        dep_refs.push_back(
            reinterpret_cast<DPCTLSyclEventRef>(const_cast<sycl::event *>(&e)));
    }

    sycl::event marker;
    const auto &api = ::dpctl::detail::dpctl_capi::get();
    if (api.SyclQueue_DeferRelease_(
            reinterpret_cast<DPCTLSyclQueueRef>(&q), dep_refs.data(),
            dep_refs.size(), release, data,
            reinterpret_cast<DPCTLSyclEventRef>(&marker)) != 0)
    {
        throw std::runtime_error("Could not defer the release");
    }

    return marker;
}

/*!
 * @brief Keeps `py_objs` alive until the work covered by `depends` completes.
 *
 * Returns an event for an empty kernel submitted to `q` after `depends`, which
 * is what the release waits for. On an in-order queue the kernel is also
 * ordered after the work already submitted there.
 *
 * `q` is intended to be the queue running the work that uses `py_objs`.
 *
 * Expects the caller to hold the GIL. Throws if the release could not be
 * scheduled, having released nothing.
 */
template <std::size_t num>
sycl::event keep_args_alive(sycl::queue &q,
                            const py::object (&py_objs)[num],
                            const std::vector<sycl::event> &depends = {})
{
    using held_t = detail::held_args<num>;

    // leaked if anything below throws, as the work in `depends` may still be
    // using what it holds
    held_t *held = new held_t{};

    for (std::size_t i = 0; i < num; ++i) {
        const auto &py_obj_i = py_objs[i];
        if (detail::ManagedMemory::is_usm_managed_by_shared_ptr(py_obj_i)) {
            held->usm_owners[held->n_usm_owners] =
                detail::ManagedMemory::extract_shared_ptr(py_obj_i);
            ++held->n_usm_owners;
        }
        else {
            held->objects[held->n_objects] = py_obj_i.inc_ref();
            ++held->n_objects;
        }
    }

    return defer_release(q, depends, &held_t::release, held);
}

/*! @brief Check if all allocation queues are the same as the
    execution queue */
template <std::size_t num>
bool queues_are_compatible(const sycl::queue &exec_q,
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
