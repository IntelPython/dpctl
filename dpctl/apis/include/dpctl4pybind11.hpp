//===----------- dpctl4pybind11.h - Headers for type pybind11 casters  -*-C-*-
//===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2025 Intel Corporation
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
#include <complex>
#include <cstddef> // for std::size_t for C++ linkage
#include <memory>
#include <pybind11/pybind11.h>
#include <stddef.h> // for size_t for C linkage
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace dpctl
{
namespace detail
{

// Lookup a type according to its size, and return a value corresponding to the
// NumPy typenum.
template <typename Concrete> constexpr int platform_typeid_lookup()
{
    return -1;
}

template <typename Concrete, typename T, typename... Ts, typename... Ints>
constexpr int platform_typeid_lookup(int I, Ints... Is)
{
    return sizeof(Concrete) == sizeof(T)
               ? I
               : platform_typeid_lookup<Concrete, Ts...>(Is...);
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
    PyTypeObject *PySyclProgramType_;
    PyTypeObject *PySyclKernelType_;

    DPCTLSyclDeviceRef (*SyclDevice_GetDeviceRef_)(PySyclDeviceObject *);
    PySyclDeviceObject *(*SyclDevice_Make_)(DPCTLSyclDeviceRef);

    DPCTLSyclContextRef (*SyclContext_GetContextRef_)(PySyclContextObject *);
    PySyclContextObject *(*SyclContext_Make_)(DPCTLSyclContextRef);

    DPCTLSyclEventRef (*SyclEvent_GetEventRef_)(PySyclEventObject *);
    PySyclEventObject *(*SyclEvent_Make_)(DPCTLSyclEventRef);

    DPCTLSyclQueueRef (*SyclQueue_GetQueueRef_)(PySyclQueueObject *);
    PySyclQueueObject *(*SyclQueue_Make_)(DPCTLSyclQueueRef);

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

    // program
    DPCTLSyclKernelRef (*SyclKernel_GetKernelRef_)(PySyclKernelObject *);
    PySyclKernelObject *(*SyclKernel_Make_)(DPCTLSyclKernelRef, const char *);

    DPCTLSyclKernelBundleRef (*SyclProgram_GetKernelBundleRef_)(
        PySyclProgramObject *);
    PySyclProgramObject *(*SyclProgram_Make_)(DPCTLSyclKernelBundleRef);

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
    bool PySyclProgram_Check_(PyObject *obj) const
    {
        return PyObject_TypeCheck(obj, PySyclProgramType_) != 0;
    }

    ~dpctl_capi()
    {
        as_usm_memory_.reset();
        default_usm_memory_.reset();
        default_sycl_queue_.reset();
    };

    static auto &get()
    {
        static dpctl_capi api{};
        return api;
    }

    py::object default_sycl_queue_pyobj() { return *default_sycl_queue_; }
    py::object default_usm_memory_pyobj() { return *default_usm_memory_; }
    py::object as_usm_memory_pyobj() { return *as_usm_memory_; }

private:
    struct Deleter
    {
        void operator()(py::object *p) const
        {
            const bool initialized = Py_IsInitialized();
#if PY_VERSION_HEX < 0x30d0000
            const bool finilizing = _Py_IsFinalizing();
#else
            const bool finilizing = Py_IsFinalizing();
#endif
            const bool guard = initialized && !finilizing;

            if (guard) {
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
          PySyclProgramType_(nullptr), PySyclKernelType_(nullptr),
          SyclDevice_GetDeviceRef_(nullptr), SyclDevice_Make_(nullptr),
          SyclContext_GetContextRef_(nullptr), SyclContext_Make_(nullptr),
          SyclEvent_GetEventRef_(nullptr), SyclEvent_Make_(nullptr),
          SyclQueue_GetQueueRef_(nullptr), SyclQueue_Make_(nullptr),
          Memory_GetUsmPointer_(nullptr), Memory_GetOpaquePointer_(nullptr),
          Memory_GetContextRef_(nullptr), Memory_GetQueueRef_(nullptr),
          Memory_GetNumBytes_(nullptr), Memory_Make_(nullptr),
          SyclKernel_GetKernelRef_(nullptr), SyclKernel_Make_(nullptr),
          SyclProgram_GetKernelBundleRef_(nullptr), SyclProgram_Make_(nullptr),
          default_sycl_queue_{}, default_usm_memory_{}, as_usm_memory_{}

    {
        // Import Cython-generated C-API for dpctl
        // This imports python modules and initializes
        // static variables such as function pointers for C-API,
        // e.g. SyclDevice_GetDeviceRef, etc.
        // and pointers to Python types, i.e. PySyclDeviceType, etc.
        import_dpctl();

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
        this->PySyclProgramType_ = &PySyclProgramType;
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

        // dpctl.memory API
        this->Memory_GetUsmPointer_ = Memory_GetUsmPointer;
        this->Memory_GetOpaquePointer_ = Memory_GetOpaquePointer;
        this->Memory_GetContextRef_ = Memory_GetContextRef;
        this->Memory_GetQueueRef_ = Memory_GetQueueRef;
        this->Memory_GetNumBytes_ = Memory_GetNumBytes;
        this->Memory_Make_ = Memory_Make;

        // dpctl.program API
        this->SyclKernel_GetKernelRef_ = SyclKernel_GetKernelRef;
        this->SyclKernel_Make_ = SyclKernel_Make;
        this->SyclProgram_GetKernelBundleRef_ = SyclProgram_GetKernelBundleRef;
        this->SyclProgram_Make_ = SyclProgram_Make;

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
 * :class:`dpctl.program.SyclKernel` for the purposes of generation of
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
                                 "dpctl.program.SyclKernel");
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

    DPCTL_TYPE_CASTER(sycl::kernel, _("dpctl.program.SyclKernel"));
};

/* This type caster associates
 * ``sycl::kernel_bundle<sycl::bundle_state::executable>`` C++ class with
 * :class:`dpctl.program.SyclProgram` for the purposes of generation of
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
        if (api.PySyclProgram_Check_(source)) {
            DPCTLSyclKernelBundleRef KBRef =
                api.SyclProgram_GetKernelBundleRef_(
                    reinterpret_cast<PySyclProgramObject *>(source));
            value = std::make_unique<
                sycl::kernel_bundle<sycl::bundle_state::executable>>(
                *(reinterpret_cast<
                    sycl::kernel_bundle<sycl::bundle_state::executable> *>(
                    KBRef)));
            return true;
        }
        else {
            throw py::type_error("Input is of unexpected type, expected "
                                 "dpctl.program.SyclProgram");
        }
    }

    static handle cast(sycl::kernel_bundle<sycl::bundle_state::executable> src,
                       return_value_policy,
                       handle)
    {
        auto const &api = ::dpctl::detail::dpctl_capi::get();
        auto tmp = api.SyclProgram_Make_(
            reinterpret_cast<DPCTLSyclKernelBundleRef>(&src));
        return handle(reinterpret_cast<PyObject *>(tmp));
    }

    DPCTL_TYPE_CASTER(sycl::kernel_bundle<sycl::bundle_state::executable>,
                      _("dpctl.program.SyclProgram"));
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

} // end of namespace detail

template <std::size_t num>
sycl::event keep_args_alive(sycl::queue &q,
                            const py::object (&py_objs)[num],
                            const std::vector<sycl::event> &depends = {})
{
    std::size_t n_objects_held = 0;
    std::array<std::shared_ptr<py::handle>, num> shp_arr{};

    std::size_t n_usm_owners_held = 0;
    std::array<std::shared_ptr<void>, num> shp_usm{};

    for (std::size_t i = 0; i < num; ++i) {
        const auto &py_obj_i = py_objs[i];
        if (detail::ManagedMemory::is_usm_managed_by_shared_ptr(py_obj_i)) {
            const auto &shp =
                detail::ManagedMemory::extract_shared_ptr(py_obj_i);
            shp_usm[n_usm_owners_held] = shp;
            ++n_usm_owners_held;
        }
        else {
            shp_arr[n_objects_held] = std::make_shared<py::handle>(py_obj_i);
            shp_arr[n_objects_held]->inc_ref();
            ++n_objects_held;
        }
    }

    bool use_depends = true;
    sycl::event host_task_ev;

    if (n_usm_owners_held > 0) {
        host_task_ev = q.submit([&](sycl::handler &cgh) {
            if (use_depends) {
                cgh.depends_on(depends);
                use_depends = false;
            }
            else {
                cgh.depends_on(host_task_ev);
            }
            cgh.host_task([shp_usm = std::move(shp_usm)]() {
                // no body, but shared pointers are captured in
                // the lambda, ensuring that USM allocation is
                // kept alive
            });
        });
    }

    if (n_objects_held > 0) {
        host_task_ev = q.submit([&](sycl::handler &cgh) {
            if (use_depends) {
                cgh.depends_on(depends);
                use_depends = false;
            }
            else {
                cgh.depends_on(host_task_ev);
            }
            cgh.host_task([n_objects_held, shp_arr = std::move(shp_arr)]() {
                py::gil_scoped_acquire acquire;

                for (std::size_t i = 0; i < n_objects_held; ++i) {
                    shp_arr[i]->dec_ref();
                }
            });
        });
    }

    return host_task_ev;
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
