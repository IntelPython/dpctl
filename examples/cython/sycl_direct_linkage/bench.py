#                      Data Parallel Control (dpCtl)
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

import dpctl
import syclbuffer_naive as sb
import numpy as np

X = np.full((10 ** 4, 4098), 1e-4, dtype="d")

# warm-up
print("=" * 10 + " Executing warm-up " + "=" * 10)
print("NumPy result: ", X.sum(axis=0))

print(
    "SYCL(default_device) result: {}".format(
        sb.columnwise_total(X),
    )
)

import timeit

print(
    "Running time of 100 calls to columnwise_total on matrix with shape {}".format(
        X.shape
    )
)

print("Times for default_selector, inclusive of queue creation:")
print(
    timeit.repeat(
        stmt="sb.columnwise_total(X)",
        setup="sb.columnwise_total(X)",  # ensure JIT compilation is not counted
        number=100,
        globals=globals(),
    )
)

print("Times for NumPy")
print(timeit.repeat(stmt="X.sum(axis=0)", number=100, globals=globals()))
