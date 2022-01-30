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

import timeit

import blackscholes_usm as bs
import numpy as np
from reference_black_scholes import ref_python_black_scholes

import dpctl
import dpctl.memory as dpctl_mem


def gen_option_params(
    n_opts, pl, ph, sl, sh, tl, th, rl, rh, vl, vh, dtype, queue=None
):
    nbytes = n_opts * 5 * np.dtype(dtype).itemsize
    usm_mem = dpctl_mem.MemoryUSMShared(nbytes, queue=queue)
    params = np.ndarray(shape=(n_opts, 5), buffer=usm_mem, dtype=dtype)
    seed = 1234
    bs.populate_params(
        params, pl, ph, sl, sh, tl, th, rl, rh, vl, vh, seed, queue=queue
    )
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

print(
    "Correctness check: allclose(Xgpu, Xref) == ",
    np.allclose(Xgpu, X_ref, atol=1e-5),
)

n_opts = 3 * 10**6

# compute on CPU sycl device

queues = []
for filter_str in ["cpu", "gpu"]:
    try:
        q = dpctl.SyclQueue(filter_str)
        queues.append(q)
    except dpctl.SyclQueueCreationError:
        continue

if not queues:
    print("No queues could not created, nothing to do.")
    exit(0)

opt_params_list = []
for q in queues:
    opt_params = gen_option_params(
        n_opts,
        20.0,
        30.0,
        22.0,
        29.0,
        18.0,
        24.0,
        0.01,
        0.05,
        0.01,
        0.05,
        "d",
        queue=q,
    )
    opt_params_list.append(opt_params)

times_dict = dict()
for q, params in zip(queues, opt_params_list):
    times_list = []
    for _ in range(5):
        t0 = timeit.default_timer()
        X1 = bs.black_scholes_price(params, queue=q)
        t1 = timeit.default_timer()
        times_list.append(t1 - t0)
    times_dict[q.name] = times_list

for dev_name, wall_times in times_dict.items():
    print("Using      : {}".format(dev_name))
    print("Wall times : {}".format(wall_times))
