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
import dpctl.memory as dpctl_mem
import blackscholes_usm as bs
import numpy as np, dpctl
from reference_black_scholes import ref_python_black_scholes


def gen_option_params(n_opts, pl, ph, sl, sh, tl, th, rl, rh, vl, vh, dtype):
    usm_mem = dpctl_mem.MemoryUSMShared(n_opts * 5 * np.dtype(dtype).itemsize)
    # usm_mem2 = dpctl_mem.MemoryUSMDevice(n_opts * 5 * np.dtype(dtype).itemsize)
    params = np.ndarray(shape=(n_opts, 5), buffer=usm_mem, dtype=dtype)
    seed = 1234
    bs.populate_params(params, pl, ph, sl, sh, tl, th, rl, rh, vl, vh, seed)
    return params


# ==== dry run ===
usm_mem = dpctl_mem.MemoryUSMShared(3 * 5 * np.dtype("d").itemsize)
opts = np.ndarray((3, 5), buffer=usm_mem, dtype="d")
# copy from Host NumPy to USM buffer
opts[:, :] = np.array(
    [
        [81.2, 81.8, 29, 0.01, 0.02],
        [24.24, 22.1, 10, 0.02, 0.08],
        [100, 100, 30, 0.01, 0.12],
    ]
)
# GPU computation
Xgpu = bs.black_scholes_price(opts)

# compute prices in CPython
X_ref = np.array([ref_python_black_scholes(*opt) for opt in opts], dtype="d")

print(np.allclose(Xgpu, X_ref, atol=1e-5))

n_opts = 3 * 10 ** 6

# compute on CPU sycl device
import timeit

for _ in range(3):

    dpctl.set_global_queue("opencl:cpu:0")
    print(
        "Using : {}".format(
            dpctl.get_current_queue().get_sycl_device().get_device_name()
        )
    )

    t0 = timeit.default_timer()
    opts1 = gen_option_params(
        n_opts, 20.0, 30.0, 22.0, 29.0, 18.0, 24.0, 0.01, 0.05, 0.01, 0.05, "d"
    )
    X1 = bs.black_scholes_price(opts1)
    t1 = timeit.default_timer()

    print("Elapsed: {}".format(t1 - t0))

    # compute on GPU sycl device
    dpctl.set_global_queue("level_zero:gpu:0")
    print(
        "Using : {}".format(
            dpctl.get_current_queue().get_sycl_device().get_device_name()
        )
    )

    t0 = timeit.default_timer()
    opts2 = gen_option_params(
        n_opts, 20.0, 30.0, 22.0, 29.0, 18.0, 24.0, 0.01, 0.05, 0.01, 0.05, "d"
    )
    X2 = bs.black_scholes_price(opts2)
    t1 = timeit.default_timer()
    print("Elapsed: {}".format(t1 - t0))

print(np.abs(opts1 - opts2).max())
print(np.abs(X2 - X1).max())
