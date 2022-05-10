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


import dpnp
import numpy as np

import dpctl
import dpctl.tensor as dpt
from dpctl import SyclTimer

n = 4000

try:
    q = dpctl.SyclQueue(property="enable_profiling")
except dpctl.SyclQueueCreationError:
    print(
        "Skipping the example, as dpctl.SyclQueue targeting "
        "default device could not be created"
    )
    exit(0)

a = dpt.reshape(dpt.arange(n * n, dtype=np.float32, sycl_queue=q), (n, n))
b = dpt.reshape(
    dpt.asarray(np.random.random(n * n), dtype=np.float32, sycl_queue=q), (n, n)
)

timer = SyclTimer(time_scale=1)

wall_times = []
device_times = []
print(
    f"Performing matrix multiplication of two {n} by {n} matrices "
    f"on {q.sycl_device.name}, repeating 5 times."
)
for _ in range(5):
    with timer(q):
        a_matmul_b = dpnp.matmul(a, b)
    host_time, device_time = timer.dt
    wall_times.append(host_time)
    device_times.append(device_time)

c = dpnp.asnumpy(a_matmul_b)
cc = np.dot(dpnp.asnumpy(a), dpnp.asnumpy(b))

print("Wall time: ", wall_times, "\nDevice time: ", device_times)
print(
    "Accuracy test: passed."
    if np.allclose(c, cc)
    else (f"Accuracy test: failed. Discrepancy {np.max(np.abs(c-cc))}")
)
