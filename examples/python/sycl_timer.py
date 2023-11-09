#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2022 Intel Corporation
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

import dpctl
import dpctl.tensor as dpt
from dpctl import SyclTimer


def matmul(m1, m2):
    """Naive matrix multiplication implementation"""
    assert m1.ndim == 2
    assert m2.ndim == 2
    assert m1.shape[1] == m2.shape[0]
    m1 = m1[:, dpt.newaxis, :]
    m2 = dpt.permute_dims(m2, (1, 0))[dpt.newaxis, :, :]
    # form m_prod[i, j, k] = m1[i,k] * m2[k, j]
    m_prods = m1 * m2
    # sum over k
    return dpt.sum(m_prods, axis=-1)


n = 500

try:
    q = dpctl.SyclQueue(property="enable_profiling")
except dpctl.SyclQueueCreationError:
    print(
        "Skipping the example, as dpctl.SyclQueue targeting "
        "default device could not be created"
    )
    exit(0)

a_flat = dpt.arange(n * n, dtype=dpt.float32, sycl_queue=q)
a = dpt.reshape(a_flat, (n, n))

b_rand = np.random.random(n * n).astype(np.float32)
b_flat = dpt.asarray(b_rand, dtype=dpt.float32, sycl_queue=q)
b = dpt.reshape(b_flat, (n, n))

wall_times = []
device_times = []

print(
    f"Computing naive matrix multiplication of two {n} by {n} matrices "
    f"on {q.sycl_device.name}, repeating 5 times."
)
print()
for _ in range(5):
    timer = SyclTimer(time_scale=1)
    with timer(q):
        a_matmul_b = matmul(a, b)
    host_time, device_time = timer.dt
    wall_times.append(host_time)
    device_times.append(device_time)

c = dpt.asnumpy(a_matmul_b)
cc = np.dot(dpt.asnumpy(a), dpt.asnumpy(b))

print("Wall time: ", wall_times, "\nDevice time: ", device_times)
print()
print(
    "Accuracy test: passed."
    if np.allclose(c, cc)
    else (f"Accuracy test: FAILED. \n   Discrepancy = {np.max(np.abs(c-cc))}")
)
