#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines unit test cases for the SyclProgram and SyclKernel classes
"""

import os

import pytest

import dpctl
import dpctl.program as dpctl_prog


def get_spirv_abspath(fn):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    spirv_file = os.path.join(curr_dir, "input_files", fn)
    return spirv_file


def _check_cpython_api_SyclProgram_GetKernelBundleRef(sycl_prog):
    """Checks Cython-generated C-API function
    `SyclProgram_GetKernelBundleRef` defined in _program.pyx"""
    import ctypes
    import sys

    assert type(sycl_prog) is dpctl_prog.SyclProgram
    mod = sys.modules[sycl_prog.__class__.__module__]
    # get capsule storing SyclProgram_GetKernelBundleRef function ptr
    kb_ref_fn_cap = mod.__pyx_capi__["SyclProgram_GetKernelBundleRef"]
    # construct Python callable to invoke "SyclProgram_GetKernelBundleRef"
    cap_ptr_fn = ctypes.pythonapi.PyCapsule_GetPointer
    cap_ptr_fn.restype = ctypes.c_void_p
    cap_ptr_fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
    kb_ref_fn_ptr = cap_ptr_fn(
        kb_ref_fn_cap,
        b"DPCTLSyclKernelBundleRef (struct PySyclProgramObject *)",
    )
    # PYFUNCTYPE(result_type, *arg_types)
    callable_maker = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object)
    get_kernel_bundle_ref_fn = callable_maker(kb_ref_fn_ptr)

    r2 = sycl_prog.addressof_ref()
    r1 = get_kernel_bundle_ref_fn(sycl_prog)
    assert r1 == r2


def _check_cpython_api_SyclProgram_Make(sycl_prog):
    """Checks Cython-generated C-API function
    `SyclProgram_Make` defined in _program.pyx"""
    import ctypes
    import sys

    assert type(sycl_prog) is dpctl_prog.SyclProgram
    mod = sys.modules[sycl_prog.__class__.__module__]
    # get capsule storing SyclProgram_Make function ptr
    make_prog_fn_cap = mod.__pyx_capi__["SyclProgram_Make"]
    # construct Python callable to invoke "SyclProgram_Make"
    cap_ptr_fn = ctypes.pythonapi.PyCapsule_GetPointer
    cap_ptr_fn.restype = ctypes.c_void_p
    cap_ptr_fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
    make_prog_fn_ptr = cap_ptr_fn(
        make_prog_fn_cap,
        b"struct PySyclProgramObject *(DPCTLSyclKernelBundleRef)",
    )
    # PYFUNCTYPE(result_type, *arg_types)
    callable_maker = ctypes.PYFUNCTYPE(ctypes.py_object, ctypes.c_void_p)
    make_prog_fn = callable_maker(make_prog_fn_ptr)

    p2 = make_prog_fn(sycl_prog.addressof_ref())
    assert p2.has_sycl_kernel("add")
    assert p2.has_sycl_kernel("axpy")


def _check_cpython_api_SyclKernel_GetKernelRef(krn):
    """Checks Cython-generated C-API function
    `SyclKernel_GetKernelRef` defined in _program.pyx"""
    import ctypes
    import sys

    assert type(krn) is dpctl_prog.SyclKernel
    mod = sys.modules[krn.__class__.__module__]
    # get capsule storing SyclKernel_GetKernelRef function ptr
    k_ref_fn_cap = mod.__pyx_capi__["SyclKernel_GetKernelRef"]
    # construct Python callable to invoke "SyclKernel_GetKernelRef"
    cap_ptr_fn = ctypes.pythonapi.PyCapsule_GetPointer
    cap_ptr_fn.restype = ctypes.c_void_p
    cap_ptr_fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
    k_ref_fn_ptr = cap_ptr_fn(
        k_ref_fn_cap, b"DPCTLSyclKernelRef (struct PySyclKernelObject *)"
    )
    # PYFUNCTYPE(result_type, *arg_types)
    callable_maker = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object)
    get_kernel_ref_fn = callable_maker(k_ref_fn_ptr)

    r2 = krn.addressof_ref()
    r1 = get_kernel_ref_fn(krn)
    assert r1 == r2


def _check_cpython_api_SyclKernel_Make(krn):
    """Checks Cython-generated C-API function
    `SyclKernel_Make` defined in _program.pyx"""
    import ctypes
    import sys

    assert type(krn) is dpctl_prog.SyclKernel
    mod = sys.modules[krn.__class__.__module__]
    # get capsule storing SyclKernel_Make function ptr
    k_make_fn_cap = mod.__pyx_capi__["SyclKernel_Make"]
    # construct Python callable to invoke "SyclKernel_Make"
    cap_ptr_fn = ctypes.pythonapi.PyCapsule_GetPointer
    cap_ptr_fn.restype = ctypes.c_void_p
    cap_ptr_fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
    k_make_fn_ptr = cap_ptr_fn(
        k_make_fn_cap,
        b"struct PySyclKernelObject *(DPCTLSyclKernelRef, char const *)",
    )
    # PYFUNCTYPE(result_type, *arg_types)
    callable_maker = ctypes.PYFUNCTYPE(
        ctypes.py_object, ctypes.c_void_p, ctypes.c_void_p
    )
    make_kernel_fn = callable_maker(k_make_fn_ptr)

    k2 = make_kernel_fn(
        krn.addressof_ref(), bytes(krn.get_function_name(), "utf-8")
    )
    assert krn.get_function_name() == k2.get_function_name()
    assert krn.get_num_args() == k2.get_num_args()
    assert krn.work_group_size == k2.work_group_size

    k3 = make_kernel_fn(krn.addressof_ref(), ctypes.c_void_p(None))
    assert k3.get_function_name() == "default_name"
    assert krn.get_num_args() == k3.get_num_args()
    assert krn.work_group_size == k3.work_group_size


def _check_multi_kernel_program(prog):
    assert type(prog) is dpctl_prog.SyclProgram

    assert type(prog.addressof_ref()) is int
    assert prog.has_sycl_kernel("add")
    assert prog.has_sycl_kernel("axpy")

    addKernel = prog.get_sycl_kernel("add")
    axpyKernel = prog.get_sycl_kernel("axpy")

    assert "add" == addKernel.get_function_name()
    assert "axpy" == axpyKernel.get_function_name()
    assert 3 == addKernel.get_num_args()
    assert 4 == axpyKernel.get_num_args()
    assert type(addKernel.addressof_ref()) is int
    assert type(axpyKernel.addressof_ref()) is int

    for krn in [addKernel, axpyKernel]:
        _check_cpython_api_SyclKernel_GetKernelRef(krn)
        _check_cpython_api_SyclKernel_Make(krn)

        na = krn.num_args
        assert na == krn.get_num_args()
        wgsz = krn.work_group_size
        assert type(wgsz) is int
        pwgszm = krn.preferred_work_group_size_multiple
        assert type(pwgszm) is int
        pmsz = krn.private_mem_size
        assert type(pmsz) is int
        vmnsg = krn.max_num_sub_groups
        assert type(vmnsg) is int
        v = krn.max_sub_group_size
        assert type(v) is int
        cmnsg = krn.compile_num_sub_groups
        assert type(cmnsg) is int
        cmsgsz = krn.compile_sub_group_size
        assert type(cmsgsz) is int

    _check_cpython_api_SyclProgram_GetKernelBundleRef(prog)
    _check_cpython_api_SyclProgram_Make(prog)


def test_create_program_from_source_ocl():
    oclSrc = "                                                             \
    kernel void add(global int* a, global int* b, global int* c) {         \
        size_t index = get_global_id(0);                                   \
        c[index] = a[index] + b[index];                                    \
    }                                                                      \
    kernel void axpy(global int* a, global int* b, global int* c, int d) { \
        size_t index = get_global_id(0);                                   \
        c[index] = a[index] + d*b[index];                                  \
    }"
    try:
        q = dpctl.SyclQueue("opencl")
    except dpctl.SyclQueueCreationError:
        pytest.skip("No OpenCL queue is available")
    prog = dpctl_prog.create_program_from_source(q, oclSrc)
    _check_multi_kernel_program(prog)


def test_create_program_from_spirv_ocl():
    try:
        q = dpctl.SyclQueue("opencl")
    except dpctl.SyclQueueCreationError:
        pytest.skip("No OpenCL queue is available")
    spirv_file = get_spirv_abspath("multi_kernel.spv")
    with open(spirv_file, "rb") as fin:
        spirv = fin.read()
    prog = dpctl_prog.create_program_from_spirv(q, spirv)
    _check_multi_kernel_program(prog)


def test_create_program_from_spirv_l0():
    try:
        q = dpctl.SyclQueue("level_zero")
    except dpctl.SyclQueueCreationError:
        pytest.skip("No Level-zero queue is available")
    spirv_file = get_spirv_abspath("multi_kernel.spv")
    with open(spirv_file, "rb") as fin:
        spirv = fin.read()
    prog = dpctl_prog.create_program_from_spirv(q, spirv)
    _check_multi_kernel_program(prog)


@pytest.mark.xfail(
    reason="Level-zero backend does not support compilation from source"
)
def test_create_program_from_source_l0():
    try:
        q = dpctl.SyclQueue("level_zero")
    except dpctl.SyclQueueCreationError:
        pytest.skip("No Level-zero queue is available")
    oclSrc = "                                                             \
    kernel void add(global int* a, global int* b, global int* c) {         \
        size_t index = get_global_id(0);                                   \
        c[index] = a[index] + b[index];                                    \
    }                                                                      \
    kernel void axpy(global int* a, global int* b, global int* c, int d) { \
        size_t index = get_global_id(0);                                   \
        c[index] = a[index] + d*b[index];                                  \
    }"
    prog = dpctl_prog.create_program_from_source(q, oclSrc)
    _check_multi_kernel_program(prog)


def test_create_program_from_invalid_src_ocl():
    try:
        q = dpctl.SyclQueue("opencl")
    except dpctl.SyclQueueCreationError:
        pytest.skip("No OpenCL queue is available")
    invalid_oclSrc = "                                                     \
    kernel void add(                                                       \
    }"
    with pytest.raises(dpctl_prog.SyclProgramCompilationError):
        dpctl_prog.create_program_from_source(q, invalid_oclSrc)
