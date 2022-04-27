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

import sys

import numpy as np

# coding: utf-8
import solve
import sycl_gemm

import dpctl
import dpctl.tensor as dpt

argv = sys.argv

n = 1000
rank = 11

if len(argv) > 1:
    n = int(argv[1])
if len(argv) > 2:
    rank = int(argv[2])


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

iters = []
for i in range(6):
    with timer(api_dev.sycl_queue):
        x, conv_in = solve.cg_solve(A, b)

    print(i, "(host_dt, device_dt)=", timer.dt)
    iters.append(conv_in)

print("Converged in: ", iters)

r = dpt.empty_like(b)
hev, ev = sycl_gemm.gemv(q, A, x, r)
delta = dpt.empty_like(b)
hev2, ev2 = sycl_gemm.sub(q, r, b, delta, [ev])
rs = sycl_gemm.norm_squared_blocking(q, delta)
dpctl.SyclEvent.wait_for([hev, hev2])
print(f"Python solution residual norm squared: {rs}")
