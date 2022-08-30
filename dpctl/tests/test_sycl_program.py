#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2021 Intel Corporation
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
        wgsz = krn.work_group_size
        assert type(wgsz) is int
        pwgszm = krn.preferred_work_group_size_multiple
        assert type(pwgszm) is int
        pmsz = krn.private_mem_size
        assert type(pmsz) is int
        vmnsg = krn.max_num_sub_groups
        assert type(vmnsg) is int
        v = krn.max_sub_group_size
        assert (
            v == NotImplemented
        ), "SyclKernel.max_sub_group_size acquired implementation, fix the test"
        cmnsg = krn.compile_num_sub_groups
        assert type(cmnsg) is int
        cmsgsz = krn.compile_num_sub_groups
        assert type(cmsgsz) is int


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
