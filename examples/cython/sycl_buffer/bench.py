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

import timeit

import numpy as np
import syclbuffer as sb

import dpctl

X = np.full((10 ** 4, 4098), 1e-4, dtype="d")

# warm-up
print("=" * 10 + " Executing warm-up " + "=" * 10)
print("NumPy result: ", X.sum(axis=0))

q = dpctl.SyclQueue("opencl:cpu")
print(
    "SYCL({}) result: {}".format(
        q.sycl_device.name,
        sb.columnwise_total(X, queue=q),
    )
)

q = dpctl.SyclQueue("opencl:gpu")
print(
    "SYCL({}) result: {}".format(
        q.sycl_device.name,
        sb.columnwise_total(X, queue=q),
    )
)


print("Times for 'opencl:cpu'")
print(
    timeit.repeat(
        stmt="sb.columnwise_total(X, queue=q)",
        setup='q = dpctl.SyclQueue("opencl:cpu"); '
        "sb.columnwise_total(X, queue=q)",  # do not count JIT compilation
        number=100,
        globals=globals(),
    )
)

print("Times for 'opencl:gpu'")
print(
    timeit.repeat(
        stmt="sb.columnwise_total(X, queue=q)",
        setup='q = dpctl.SyclQueue("opencl:gpu"); '
        "sb.columnwise_total(X, queue=q)",
        number=100,
        globals=globals(),
    )
)

print("Times for NumPy")
print(timeit.repeat(stmt="X.sum(axis=0)", number=100, globals=globals()))
