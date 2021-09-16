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

# coding: utf-8

import numpy as np
import pybind11_example as eg

import dpctl

q = dpctl.SyclQueue()

# Pass dpctl.SyclQueue to Pybind11 extension
eu_count = eg.get_max_compute_units(q)
global_mem_size = eg.get_device_global_mem_size(q.sycl_device)
local_mem_size = eg.get_device_local_mem_size(q.sycl_device)

print(f"EU count returned by Pybind11 extension {eu_count}")
print("EU count computed by dpctl {}".format(q.sycl_device.max_compute_units))
print("Device's global memory size:  {} bytes".format(global_mem_size))
print("Device's local memory size:  {} bytes".format(local_mem_size))

print("")
print("Computing modular reduction using SYCL on a NumPy array")

X = np.random.randint(low=1, high=2 ** 16 - 1, size=10 ** 6, dtype=np.longlong)
modulus_p = 347

Y = eg.offloaded_array_mod(
    q, X, modulus_p
)  # Y is a regular array with host memory underneath it
Ynp = X % modulus_p

check = np.array_equal(Y, Ynp)

if check:
    print("Offloaded result agrees with reference one computed by NumPy")
else:
    print("Offloaded array differs from reference result computed by NumPy")
