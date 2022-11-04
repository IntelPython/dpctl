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

struct dpctl_capi
{

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
    PyTypeObject *PyUSMArrayType_;
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
    DPCTLSyclContextRef (*Memory_GetContextRef_)(Py_MemoryObject *);
    DPCTLSyclQueueRef (*Memory_GetQueueRef_)(Py_MemoryObject *);
    size_t (*Memory_GetNumBytes_)(Py_MemoryObject *);
    PyObject *(*Memory_Make_)(DPCTLSyclUSMRef,
                              size_t,
                              DPCTLSyclQueueRef,
                              PyObject *);

    // program
    DPCTLSyclKernelRef (*SyclKernel_GetKernelRef_)(PySyclKernelObject *);
    PySyclKernelObject *(*SyclKernel_Make_)(DPCTLSyclKernelRef);

    DPCTLSyclKernelBundleRef (*SyclProgram_GetKernelBundleRef_)(PySyclProgramObject *);
    PySyclProgramObject *(*SyclProgram_Make_)(DPCTLSyclKernelBundleRef);

    // tensor
    char *(*UsmNDArray_GetData_)(PyUSMArrayObject *);
    int (*UsmNDArray_GetNDim_)(PyUSMArrayObject *);
    py::ssize_t *(*UsmNDArray_GetShape_)(PyUSMArrayObject *);
    py::ssize_t *(*UsmNDArray_GetStrides_)(PyUSMArrayObject *);
    int (*UsmNDArray_GetTypenum_)(PyUSMArrayObject *);
    int (*UsmNDArray_GetElementSize_)(PyUSMArrayObject *);
    int (*UsmNDArray_GetFlags_)(PyUSMArrayObject *);
    DPCTLSyclQueueRef (*UsmNDArray_GetQueueRef_)(PyUSMArrayObject *);
    py::ssize_t (*UsmNDArray_GetOffset_)(PyUSMArrayObject *);

    int USM_ARRAY_C_CONTIGUOUS_;
    int USM_ARRAY_F_CONTIGUOUS_;
    int USM_ARRAY_WRITABLE_;
    int UAR_BOOL_, UAR_BYTE_, UAR_UBYTE_, UAR_SHORT_, UAR_USHORT_, UAR_INT_,
        UAR_UINT_, UAR_LONG_, UAR_ULONG_, UAR_LONGLONG_, UAR_ULONGLONG_,
        UAR_FLOAT_, UAR_DOUBLE_, UAR_CFLOAT_, UAR_CDOUBLE_, UAR_TYPE_SENTINEL_,
        UAR_HALF_;
    int UAR_INT8_, UAR_UINT8_, UAR_INT16_, UAR_UINT16_, UAR_INT32_, UAR_UINT32_,
        UAR_INT64_, UAR_UINT64_;

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

    ~dpctl_capi(){};

    static auto &get()
    {
        static dpctl_capi api = lookup();
        return api;
    }

    py::object default_sycl_queue_pyobj()
    {
        return *default_sycl_queue;
    }
    py::object default_usm_memory_pyobj()
    {
        return *default_usm_memory;
    }
    py::object default_usm_ndarray_pyobj()
    {
        return *default_usm_ndarray;
    }
    py::object as_usm_memory_pyobj()
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

    std::shared_ptr<py::object> default_sycl_queue;
    std::shared_ptr<py::object> default_usm_memory;
    std::shared_ptr<py::object> default_usm_ndarray;
    std::shared_ptr<py::object> as_usm_memory;
    std::shared_ptr<py::object> default_sycl_kernel;
    std::shared_ptr<py::object> default_sycl_program;

    dpctl_capi()
        : default_sycl_queue{}, default_usm_memory{}, default_usm_ndarray{},
          as_usm_memory{}
    {
        // Import Cython-generated C-API for dpctl
        // This imports python modules and initializes
        // static variables such as function pointers for C-API,
        // e.g. SyclDevice_GetDeviceRef, etc.
        // pointers to Python types, i.e. PySyclDeviceType, etc.
        // and exported constants, i.e. USM_ARRAY_C_CONTIGUOUS, etc.
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
        this->PyUSMArrayType_ = &PyUSMArrayType;
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
        this->Memory_GetContextRef_ = Memory_GetContextRef;
        this->Memory_GetQueueRef_ = Memory_GetQueueRef;
        this->Memory_GetNumBytes_ = Memory_GetNumBytes;
        this->Memory_Make_ = Memory_Make;

        // dpctl.program API
        this->SyclKernel_Make_ = SyclKernel_Make;
        this->SyclProgram_Make_ = SyclProgram_Make;

        // dpctl.tensor.usm_ndarray API
        this->UsmNDArray_GetData_ = UsmNDArray_GetData;
        this->UsmNDArray_GetNDim_ = UsmNDArray_GetNDim;
        this->UsmNDArray_GetShape_ = UsmNDArray_GetShape;
        this->UsmNDArray_GetStrides_ = UsmNDArray_GetStrides;
        this->UsmNDArray_GetTypenum_ = UsmNDArray_GetTypenum;
        this->UsmNDArray_GetElementSize_ = UsmNDArray_GetElementSize;
        this->UsmNDArray_GetFlags_ = UsmNDArray_GetFlags;
        this->UsmNDArray_GetQueueRef_ = UsmNDArray_GetQueueRef;
        this->UsmNDArray_GetOffset_ = UsmNDArray_GetOffset;

        // constants
        this->USM_ARRAY_C_CONTIGUOUS_ = USM_ARRAY_C_CONTIGUOUS;
        this->USM_ARRAY_F_CONTIGUOUS_ = USM_ARRAY_F_CONTIGUOUS;
        this->USM_ARRAY_WRITABLE_ = USM_ARRAY_WRITABLE;
        this->UAR_BOOL_ = UAR_BOOL;
        this->UAR_SHORT_ = UAR_SHORT;
        this->UAR_USHORT_ = UAR_USHORT;
        this->UAR_INT_ = UAR_INT;
        this->UAR_UINT_ = UAR_UINT;
        this->UAR_LONG_ = UAR_LONG;
        this->UAR_ULONG_ = UAR_ULONG;
        this->UAR_LONGLONG_ = UAR_LONGLONG;
        this->UAR_ULONGLONG_ = UAR_ULONGLONG;
        this->UAR_FLOAT_ = UAR_FLOAT;
        this->UAR_DOUBLE_ = UAR_DOUBLE;
        this->UAR_CFLOAT_ = UAR_CFLOAT;
        this->UAR_CDOUBLE_ = UAR_CDOUBLE;
        this->UAR_TYPE_SENTINEL_ = UAR_TYPE_SENTINEL;
        this->UAR_HALF_ = UAR_HALF;

        // deduced disjoint types
        this->UAR_INT8_ = UAR_BYTE;
        this->UAR_UINT8_ = UAR_UBYTE;
        this->UAR_INT16_ = UAR_SHORT;
        this->UAR_UINT16_ = UAR_USHORT;
        this->UAR_INT32_ =
            platform_typeid_lookup<std::int32_t, long, int, short>(
                UAR_LONG, UAR_INT, UAR_SHORT);
        this->UAR_UINT32_ =
            platform_typeid_lookup<std::uint32_t, unsigned long, unsigned int,
                                   unsigned short>(UAR_ULONG, UAR_UINT,
                                                   UAR_USHORT);
        this->UAR_INT64_ =
            platform_typeid_lookup<std::int64_t, long, long long, int>(
                UAR_LONG, UAR_LONGLONG, UAR_INT);
        this->UAR_UINT64_ =
            platform_typeid_lookup<std::uint64_t, unsigned long,
                                   unsigned long long, unsigned int>(
                UAR_ULONG, UAR_ULONGLONG, UAR_UINT);

        // create shared pointers to python objects used in type-casters
        // for dpctl::memory::usm_memory and dpctl::tensor::usm_ndarray
        sycl::queue q_{};
        PySyclQueueObject *py_q_tmp =
            SyclQueue_Make(reinterpret_cast<DPCTLSyclQueueRef>(&q_));
        py::object py_sycl_queue = py::reinterpret_steal<py::object>(
            reinterpret_cast<PyObject *>(py_q_tmp));

        default_sycl_queue = std::shared_ptr<py::object>(
            new py::object(py_sycl_queue), Deleter{});

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

    dpctl_capi(dpctl_capi const &) = default;
    dpctl_capi &operator=(dpctl_capi const &) = default;

    static dpctl_capi lookup()
    {
        static dpctl_capi api;
        return api;
    }

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
        auto &api = ::dpctl::detail::dpctl_capi::get();
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
        auto &api = ::dpctl::detail::dpctl_capi::get();
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
        auto &api = ::dpctl::detail::dpctl_capi::get();
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
        auto &api = ::dpctl::detail::dpctl_capi::get();
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
        auto &api = ::dpctl::detail::dpctl_capi::get();
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
        auto &api = ::dpctl::detail::dpctl_capi::get();
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
        auto &api = ::dpctl::detail::dpctl_capi::get();
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
        auto &api = ::dpctl::detail::dpctl_capi::get();
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
        auto &api = ::dpctl::detail::dpctl_capi::get();
        if (api.PySyclKernel_Check_(source)) {
            DPCTLSyclKernelRef KRef = api.SyclKernel_GetKernelRef_(
                reinterpret_cast<PySyclKernelObject *>(source));
            value = std::make_unique<sycl::kernel>(
                *(reinterpret_cast<sycl::kernel *>(KRef)));
            return true;
        }
        else {
            throw py::type_error(
                "Input is of unexpected type, expected dpctl.program.SyclKernel");
        }
    }

    static handle cast(sycl::kernel src, return_value_policy, handle)
    {
        auto &api = ::dpctl::detail::dpctl_capi::get();
        auto tmp =
            api.SyclKernel_Make_(reinterpret_cast<DPCTLSyclKernelRef>(&src));
        return handle(reinterpret_cast<PyObject *>(tmp));
    }

    DPCTL_TYPE_CASTER(sycl::kernel, _("dpctl.program.SyclKernel"));
};

/* This type caster associates ``sycl::kernel_bundle<sycl::bundle_state::executable>`` C++ class with
 * :class:`dpctl.program.SyclProgram` for the purposes of generation of
 * Python bindings by pybind11.
 */
template <> struct type_caster<sycl::kernel_bundle<sycl::bundle_state::executable>>
{
public:
    bool load(handle src, bool)
    {
        PyObject *source = src.ptr();
        auto &api = ::dpctl::detail::dpctl_capi::get();
        if (api.PySyclProgram_Check_(source)) {
            DPCTLSyclKernelBundleRef KBRef = api.SyclProgram_GetKernelBundleRef_(
                reinterpret_cast<PySyclProgramObject *>(source));
            value = std::make_unique<sycl::kernel_bundle<sycl::bundle_state::executable>>(
                *(reinterpret_cast<sycl::kernel_bundle<sycl::bundle_state::executable> *>(KBRef)));
            return true;
        }
        else {
            throw py::type_error(
                "Input is of unexpected type, expected dpctl.SyclEvent");
        }
    }

    static handle cast(sycl::kernel_bundle<sycl::bundle_state::executable> src, return_value_policy, handle)
    {
        auto &api = ::dpctl::detail::dpctl_capi::get();
        auto tmp =
            api.SyclProgram_Make_(reinterpret_cast<DPCTLSyclKernelBundleRef>(&src));
        return handle(reinterpret_cast<PyObject *>(tmp));
    }

    DPCTL_TYPE_CASTER(sycl::kernel_bundle<sycl::bundle_state::executable>, _("dpctl.program.SyclProgram"));
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

    sycl::queue get_queue() const
    {
        Py_MemoryObject *mem_obj = reinterpret_cast<Py_MemoryObject *>(m_ptr);
        auto &api = ::dpctl::detail::dpctl_capi::get();
        DPCTLSyclQueueRef QRef = api.Memory_GetQueueRef_(mem_obj);
        sycl::queue *obj_q = reinterpret_cast<sycl::queue *>(QRef);
        return *obj_q;
    }

    char *get_pointer() const
    {
        Py_MemoryObject *mem_obj = reinterpret_cast<Py_MemoryObject *>(m_ptr);
        auto &api = ::dpctl::detail::dpctl_capi::get();
        DPCTLSyclUSMRef MRef = api.Memory_GetUsmPointer_(mem_obj);
        return reinterpret_cast<char *>(MRef);
    }

    size_t get_nbytes() const
    {
        auto &api = ::dpctl::detail::dpctl_capi::get();
        Py_MemoryObject *mem_obj = reinterpret_cast<Py_MemoryObject *>(m_ptr);
        return api.Memory_GetNumBytes_(mem_obj);
    }

protected:
    static PyObject *as_usm_memory(PyObject *o)
    {
        if (o == nullptr) {
            PyErr_SetString(PyExc_ValueError,
                            "cannot create a usm_memory from a nullptr");
            return nullptr;
        }

        auto convertor =
            ::dpctl::detail::dpctl_capi::get().as_usm_memory_pyobj();

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

inline std::vector<py::ssize_t>
c_contiguous_strides(int nd,
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

inline std::vector<py::ssize_t>
f_contiguous_strides(int nd,
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

inline std::vector<py::ssize_t>
c_contiguous_strides(const std::vector<py::ssize_t> &shape,
                     py::ssize_t element_size = 1)
{
    return c_contiguous_strides(shape.size(), shape.data(), element_size);
}

inline std::vector<py::ssize_t>
f_contiguous_strides(const std::vector<py::ssize_t> &shape,
                     py::ssize_t element_size = 1)
{
    return f_contiguous_strides(shape.size(), shape.data(), element_size);
}

class usm_ndarray : public py::object
{
public:
    PYBIND11_OBJECT(usm_ndarray, py::object, [](PyObject *o) -> bool {
        return PyObject_TypeCheck(
                   o, ::dpctl::detail::dpctl_capi::get().PyUSMArrayType_) != 0;
    })

    usm_ndarray()
        : py::object(
              ::dpctl::detail::dpctl_capi::get().default_usm_ndarray_pyobj(),
              borrowed_t{})
    {
        if (!m_ptr)
            throw py::error_already_set();
    }

    char *get_data() const
    {
        PyUSMArrayObject *raw_ar = this->usm_array_ptr();

        auto &api = ::dpctl::detail::dpctl_capi::get();
        return api.UsmNDArray_GetData_(raw_ar);
    }

    template <typename T> T *get_data() const
    {
        return reinterpret_cast<T *>(get_data());
    }

    int get_ndim() const
    {
        PyUSMArrayObject *raw_ar = this->usm_array_ptr();

        auto &api = ::dpctl::detail::dpctl_capi::get();
        return api.UsmNDArray_GetNDim_(raw_ar);
    }

    const py::ssize_t *get_shape_raw() const
    {
        PyUSMArrayObject *raw_ar = this->usm_array_ptr();

        auto &api = ::dpctl::detail::dpctl_capi::get();
        return api.UsmNDArray_GetShape_(raw_ar);
    }

    py::ssize_t get_shape(int i) const
    {
        auto shape_ptr = get_shape_raw();
        return shape_ptr[i];
    }

    const py::ssize_t *get_strides_raw() const
    {
        PyUSMArrayObject *raw_ar = this->usm_array_ptr();

        auto &api = ::dpctl::detail::dpctl_capi::get();
        return api.UsmNDArray_GetStrides_(raw_ar);
    }

    py::ssize_t get_size() const
    {
        PyUSMArrayObject *raw_ar = this->usm_array_ptr();

        auto &api = ::dpctl::detail::dpctl_capi::get();
        int ndim = api.UsmNDArray_GetNDim_(raw_ar);
        const py::ssize_t *shape = api.UsmNDArray_GetShape_(raw_ar);

        py::ssize_t nelems = 1;
        for (int i = 0; i < ndim; ++i) {
            nelems *= shape[i];
        }

        assert(nelems >= 0);
        return nelems;
    }

    std::pair<py::ssize_t, py::ssize_t> get_minmax_offsets() const
    {
        PyUSMArrayObject *raw_ar = this->usm_array_ptr();

        auto &api = ::dpctl::detail::dpctl_capi::get();
        int nd = api.UsmNDArray_GetNDim_(raw_ar);
        const py::ssize_t *shape = api.UsmNDArray_GetShape_(raw_ar);
        const py::ssize_t *strides = api.UsmNDArray_GetStrides_(raw_ar);

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
            offset_min = api.UsmNDArray_GetOffset_(raw_ar);
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
        PyUSMArrayObject *raw_ar = this->usm_array_ptr();

        auto &api = ::dpctl::detail::dpctl_capi::get();
        DPCTLSyclQueueRef QRef = api.UsmNDArray_GetQueueRef_(raw_ar);
        return *(reinterpret_cast<sycl::queue *>(QRef));
    }

    int get_typenum() const
    {
        PyUSMArrayObject *raw_ar = this->usm_array_ptr();

        auto &api = ::dpctl::detail::dpctl_capi::get();
        return api.UsmNDArray_GetTypenum_(raw_ar);
    }

    int get_flags() const
    {
        PyUSMArrayObject *raw_ar = this->usm_array_ptr();

        auto &api = ::dpctl::detail::dpctl_capi::get();
        return api.UsmNDArray_GetFlags_(raw_ar);
    }

    int get_elemsize() const
    {
        PyUSMArrayObject *raw_ar = this->usm_array_ptr();

        auto &api = ::dpctl::detail::dpctl_capi::get();
        return api.UsmNDArray_GetElementSize_(raw_ar);
    }

    bool is_c_contiguous() const
    {
        int flags = this->get_flags();
        auto &api = ::dpctl::detail::dpctl_capi::get();
        return static_cast<bool>(flags & api.USM_ARRAY_C_CONTIGUOUS_);
    }

    bool is_f_contiguous() const
    {
        int flags = this->get_flags();
        auto &api = ::dpctl::detail::dpctl_capi::get();
        return static_cast<bool>(flags & api.USM_ARRAY_F_CONTIGUOUS_);
    }

    bool is_writable() const
    {
        int flags = this->get_flags();
        auto &api = ::dpctl::detail::dpctl_capi::get();
        return static_cast<bool>(flags & api.USM_ARRAY_WRITABLE_);
    }

private:
    PyUSMArrayObject *usm_array_ptr() const
    {
        return reinterpret_cast<PyUSMArrayObject *>(m_ptr);
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
