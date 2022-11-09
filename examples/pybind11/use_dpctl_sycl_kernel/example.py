#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2022 Intel Corporation
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

import use_kernel as eg

import dpctl
import dpctl.program as dppr
import dpctl.tensor as dpt

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
x = dpt.arange(0, stop=13, step=1, dtype="i4", sycl_queue=q)
y = dpt.empty_like(x)

eg.submit_custom_kernel(q, krn, src=x, dst=y)

# output the result
print(dpt.asnumpy(y))
