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

import timeit

import numpy as np
import syclbuffer as sb

import dpctl


class Skipped:
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return f"Skipped(reason='{self.msg}')"

    def __repr__(self):
        return self.__str__()


def bench_offload(selector_string, X):
    try:
        q = dpctl.SyclQueue(selector_string)
    except dpctl.SyclQueueCreationError:
        return Skipped(
            f"Skipping run for {selector_string}, queue could nor be created"
        )
    return timeit.repeat(
        stmt="sb.columnwise_total(X, queue=q)",
        setup="q = dpctl.SyclQueue(selector_string); "
        "sb.columnwise_total(X, queue=q)",  # do not count JIT compilation
        number=100,
        globals={
            "q": q,
            "X": X,
            "dpctl": dpctl,
            "sb": sb,
            "selector_string": selector_string,
        },
    )


def run_offload(selector_string, X):
    try:
        q = dpctl.SyclQueue(selector_string)
    except dpctl.SyclQueueCreationError:
        return Skipped(
            f"Skipping run for {selector_string}, queue could nor be created"
        )
    return (
        f"SYCL({q.sycl_device.name}) result: "
        f"{sb.columnwise_total(X, queue=q)}"
    )


X = np.full((10**6, 15), 1e-4, dtype="f4")

print(f"Matrix size: {X.shape}, dtype = {X.dtype}")

# warm-up
print("=" * 10 + " Executing warm-up " + "=" * 10)
print("NumPy result: ", X.sum(axis=0))

for ss in ["opencl:cpu", "opencl:gpu", "level_zero:gpu"]:
    print(f"Result for '{ss}': {run_offload(ss, X)}")

print("=" * 10 + " Running benchmarks " + "=" * 10)

for ss in ["opencl:cpu", "opencl:gpu", "level_zero:gpu"]:
    print(f"Timing offload to '{ss}': {bench_offload(ss, X)}")


print(
    "Times for NumPy: "
    f"{timeit.repeat(stmt='X.sum(axis=0)', number=100, globals=globals())}"
)
