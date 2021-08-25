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


import numba_dppy
import numpy as np
from sycl_timer import SyclTimer

import dpctl


@numba_dppy.kernel
def dppy_gemm(a, b, c):
    i = numba_dppy.get_global_id(0)
    j = numba_dppy.get_global_id(1)
    if i >= c.shape[0] or j >= c.shape[1]:
        return
    c[i, j] = 0
    for k in range(c.shape[0]):
        c[i, j] += a[i, k] * b[k, j]


X = 1024
Y = 16
global_size = X, X

griddim = X, X
blockdim = Y, Y

a = np.arange(X * X, dtype=np.float32).reshape(X, X)
b = np.array(np.random.random(X * X), dtype=np.float32).reshape(X, X)
c = np.ones_like(a).reshape(X, X)

q = dpctl.SyclQueue("opencl:gpu", property="enable_profiling")
with dpctl.device_context(q):
    timers = SyclTimer(time_scale=1)
    with timers(q):
        dppy_gemm[griddim, blockdim](a, b, c)
        cc = np.dot(a, b)
    host_time, device_time = timers.dt()
    print("Wall time: ", host_time, "\n", "Device time: ", device_time)
    print(np.allclose(c, cc))
