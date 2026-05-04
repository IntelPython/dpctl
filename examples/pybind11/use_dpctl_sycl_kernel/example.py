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

# coding: utf-8

import numpy as np
import use_kernel as eg

import dpctl
import dpctl.memory as dpmem
import dpctl.program as dppr

# create execution queue, targeting default selected device
q = dpctl.SyclQueue()

# read SPIR-V: a program in Khronos standardized intermediate form
with open("resource/double_it.spv", "br") as fh:
    il = fh.read()

# Build the program for the selected device
pr = dppr.create_program_from_spirv(q, il, "")
assert pr.has_sycl_kernel("double_it")

# Retrieve the kernel from the problem
krn = pr.get_sycl_kernel("double_it")
assert krn.num_args == 2

# Construct the argument, and allocate memory for the result
x = np.arange(0, stop=13, step=1, dtype="i4")
y = np.empty_like(x)
x_dev = dpmem.MemoryUSMDevice(x.nbytes, queue=q)
y_dev = dpmem.MemoryUSMDevice(y.nbytes, queue=q)

# Copy input data to the device
q.memcpy(dest=x_dev, src=x, count=x.nbytes)

eg.submit_custom_kernel(q, krn, src=x_dev, dst=y_dev)

# Copy result data back to host
q.memcpy(dest=y, src=y_dev, count=y.nbytes)

# output the result
print(y)
