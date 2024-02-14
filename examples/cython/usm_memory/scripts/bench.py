#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2024 Intel Corporation
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

import blackscholes as bs

import dpctl
import dpctl.tensor as dpt


def gen_option_params(
    n_opts, pl, ph, sl, sh, tl, th, rl, rh, vl, vh, dtype, queue=None
):
    params = dpt.empty((n_opts, 5), dtype=dtype, sycl_queue=queue)
    seed = 1234
    bs.populate_params(params, pl, ph, sl, sh, tl, th, rl, rh, vl, vh, seed)
    return params


n_opts = 3 * 10**7

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
        "f",
        queue=q,
    )
    opt_params_list.append(opt_params)

times_dict = dict()
dtype_dict = dict()

for q, params in zip(queues, opt_params_list):
    times_list = []
    for _ in range(5):
        t0 = timeit.default_timer()
        X1 = bs.black_scholes_price(params)
        t1 = timeit.default_timer()
        times_list.append(t1 - t0)
    times_dict[q.name] = times_list
    dtype_dict[q.name] = params.dtype

print(
    f"Pricing {n_opts:,} vanilla European options using "
    "Black-Scholes-Merton formula"
)
print("")
for dev_name, wall_times in times_dict.items():
    print("Using      : {}".format(dev_name))
    print(
        "Wall times : {} for dtype={}".format(wall_times, dtype_dict[dev_name])
    )
