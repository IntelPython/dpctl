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

import numpy as np

# coding: utf-8
import solve

import dpctl
import dpctl.tensor as dpt

n = 1000
rank = 16

print(
    f"Solving {n} by {n} diagonal linear "
    f"system with rank {rank} perturbation."
)

Anp = np.eye(n, n) + (lambda x: x.T @ x)(np.random.randn(rank, n))
bnp = np.random.rand(n)

q = dpctl.SyclQueue(property=["enable_profiling"])
q.print_device_info()
if q.is_in_order:
    print("Using in-order queue")
else:
    print("Using not in-order queue")

api_dev = dpctl.tensor.Device.create_device(q)
A = dpt.asarray(Anp, "d", device=api_dev)
b = dpt.asarray(bnp, "d", device=api_dev)

timer = dpctl.SyclTimer(time_scale=1e3)

for i in range(20):
    with timer(api_dev.sycl_queue):
        solve.cg_solve(A, b)

    print(i, timer.dt)
