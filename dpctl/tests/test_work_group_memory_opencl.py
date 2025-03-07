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
import dpctl.tensor

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

    x_dev = dpctl.memory.MemoryUSMDevice(global_size * 4, queue=q)
    y_dev = dpctl.memory.MemoryUSMDevice(global_size * 4, queue=q)

    x = np.ones(global_size, dtype="float32")
    y = np.zeros(global_size, dtype="float32")
    q.memcpy(x_dev, x, x_dev.nbytes)
    q.memcpy(y_dev, y, y_dev.nbytes)

    try:
        q.submit(
            kernel,
            [
                x_dev,
                y_dev,
                dpctl.WorkGroupMemory(local_size * x.itemsize),
            ],
            [global_size],
            [local_size],
        )
        q.wait()
    except dpctl._sycl_queue.SyclKernelSubmitError:
        pytest.fail("Foo")
        pytest.skip(f"Kernel submission to {q.sycl_device} failed")

    q.memcpy(y, y_dev, y_dev.nbytes)

    assert np.all(x == y)
