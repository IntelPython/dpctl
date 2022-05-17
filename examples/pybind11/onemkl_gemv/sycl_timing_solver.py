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

assert A.sycl_queue == b.sycl_queue

# allocate buffers for computation of residual
r = dpt.empty_like(b)
delta = dpt.empty_like(b)

timer = dpctl.SyclTimer(time_scale=1e3)


def time_python_solver(num_iters=6):
    """
    Time solver implemented in Python with use of asynchronous
    SYCL kernel submission.
    """
    global x
    iters = []
    for i in range(num_iters):
        with timer(api_dev.sycl_queue):
            x, conv_in = solve.cg_solve(A, b)

        print(i, "(host_dt, device_dt)=", timer.dt)
        iters.append(conv_in)
        assert x.usm_type == A.usm_type
        assert x.usm_type == b.usm_type
        assert x.sycl_queue == A.sycl_queue
        assert x.sycl_queue == b.sycl_queue

    return iters


def time_cpp_solver(num_iters=6):
    """
    Time solver implemented in C++ but callable from Python.
    C++ implementation uses the same algorithm and submits same
    kernels asynchronously, but bypasses Python binding overhead
    incurred when algorithm is driver from Python.
    """
    global x_cpp
    x_cpp = dpt.empty_like(b)
    iters = []
    for i in range(num_iters):
        with timer(api_dev.sycl_queue):
            conv_in = sycl_gemm.cpp_cg_solve(q, A, b, x_cpp)

        print(i, "(host_dt, device_dt)=", timer.dt)
        iters.append(conv_in)

    return iters


def compute_residual(x):
    """
    Computes quality of the solution, `norm_squared(A@x - b)`.
    """
    assert isinstance(x, dpt.usm_ndarray)
    q = A.sycl_queue
    hev, ev = sycl_gemm.gemv(q, A, x, r)
    hev2, ev2 = sycl_gemm.sub(q, r, b, delta, [ev])
    rs = sycl_gemm.norm_squared_blocking(q, delta)
    dpctl.SyclEvent.wait_for([hev, hev2])
    return rs


print("Converged in: ", time_python_solver())
print(f"Python solution residual norm squared: {compute_residual(x)}")

assert q == api_dev.sycl_queue
print("")

print("Converged in: ", time_cpp_solver())
print(f"cpp_cg_solve solution residual norm squared: {compute_residual(x_cpp)}")
