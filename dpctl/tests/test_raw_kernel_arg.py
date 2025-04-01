#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2025 Intel Corporation
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

"""Defines unit test cases for the work_group_memory in a SYCL kernel"""

import ctypes
import os

import pytest

import dpctl
import dpctl.tensor


def get_spirv_abspath(fn):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    spirv_file = os.path.join(curr_dir, "input_files", fn)
    return spirv_file


# The kernel in the SPIR-V file used in this test was generated from the
# following SYCL source code:
# #include <sycl/sycl.hpp>
#
# using namespace sycl;
#
# namespace syclexp = sycl::ext::oneapi::experimental;
# namespace syclext = sycl::ext::oneapi;
#
# using data_t = int32_t;
#
# struct Params { data_t mul; data_t add; };
#
# extern "C" SYCL_EXTERNAL
# SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
# void raw_arg_kernel(data_t* in, data_t* out, Params p){
#   auto item = syclext::this_work_item::get_nd_item<1>();
#   size_t global_id = item.get_global_linear_id();
#   out[global_id] = (in[global_id] * p.mul) + p.add;
# }


class Params(ctypes.Structure):
    _fields_ = [("mul", ctypes.c_int32), ("add", ctypes.c_int32)]


def launch_raw_arg_kernel(raw):
    if not dpctl.RawKernelArg.is_available():
        pytest.skip("Raw kernel arg extension not supported")

    try:
        q = dpctl.SyclQueue("level_zero")
    except dpctl.SyclQueueCreationError:
        pytest.skip("LevelZero queue could not be created")
    spirv_file = get_spirv_abspath("raw-arg-kernel.spv")
    with open(spirv_file, "br") as spv:
        spv_bytes = spv.read()
    prog = dpctl.program.create_program_from_spirv(q, spv_bytes)
    kernel = prog.get_sycl_kernel("__sycl_kernel_raw_arg_kernel")
    local_size = 16
    global_size = local_size * 8

    x = dpctl.tensor.ones(global_size, dtype="int32")
    y = dpctl.tensor.zeros(global_size, dtype="int32")
    x.sycl_queue.wait()
    y.sycl_queue.wait()

    try:
        q.submit(
            kernel,
            [
                x.usm_data,
                y.usm_data,
                raw,
            ],
            [global_size],
            [local_size],
        )
        q.wait()
    except dpctl._sycl_queue.SyclKernelSubmitError:
        pytest.skip(f"Kernel submission to {q.sycl_device} failed")

    assert dpctl.tensor.all(y == 9)


def test_submit_raw_kernel_arg_pointer():
    paramStruct = Params(4, 5)
    raw = dpctl.RawKernelArg(
        ctypes.sizeof(paramStruct), ctypes.addressof(paramStruct)
    )
    launch_raw_arg_kernel(raw)


def test_submit_raw_kernel_arg_buffer():
    paramStruct = Params(4, 5)
    byteArr = bytearray(paramStruct)
    raw = dpctl.RawKernelArg(byteArr)
    del byteArr
    launch_raw_arg_kernel(raw)
