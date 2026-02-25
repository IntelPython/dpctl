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

"""Defines unit test cases for the work_group_memory in an OpenCL kernel"""

import numpy as np
import pytest

import dpctl

ocl_kernel_src = """
__kernel void local_mem_kernel(__global float *input, __global float *output,
                                __local float *local_data) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);

    // Load input data into local memory
    local_data[lid] = input[gid];

    // Store the data in the output array
    output[gid] = local_data[lid];
}
"""


def test_submit_work_group_memory_opencl():
    if not dpctl.WorkGroupMemory.is_available():
        pytest.skip("Work group memory extension not supported")

    try:
        q = dpctl.SyclQueue("opencl")
    except dpctl.SyclQueueCreationError:
        pytest.skip("OpenCL queue could not be created")

    prog = dpctl.program.create_program_from_source(q, ocl_kernel_src)
    kernel = prog.get_sycl_kernel("local_mem_kernel")
    local_size = 16
    global_size = local_size * 8

    x = np.ones(global_size, dtype="float32")
    y = np.zeros(global_size, dtype="float32")

    x_usm = dpctl.memory.MemoryUSMDevice(x.nbytes, queue=q)
    y_usm = dpctl.memory.MemoryUSMDevice(y.nbytes, queue=q)

    ev1 = q.memcpy_async(dest=x_usm, src=x, count=x.nbytes)

    try:
        ev2 = q.submit(
            kernel,
            [
                x_usm,
                y_usm,
                dpctl.WorkGroupMemory(local_size * x.itemsize),
            ],
            [global_size],
            [local_size],
            dEvents=[ev1],
        )
    except dpctl._sycl_queue.SyclKernelSubmitError:
        pytest.skip(f"Kernel submission to {q.sycl_device} failed")

    ev3 = q.memcpy_async(dest=y, src=y_usm, count=y.nbytes, dEvents=[ev2])
    ev3.wait()

    assert np.all(x == y)
