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

import numpy as np
import use_queue_device as uqd

import dpctl


def test_compute_units():
    q = dpctl.SyclQueue()
    mcu = uqd.get_max_compute_units(q)

    assert type(mcu) is int
    assert mcu == q.sycl_device.max_compute_units


def test_global_memory():
    d = dpctl.SyclDevice()
    gm = uqd.get_device_global_mem_size(d)
    assert type(gm) is int
    assert gm == d.global_mem_size


def test_local_memory():
    d = dpctl.SyclDevice()
    lm = uqd.get_device_local_mem_size(d)
    assert type(lm) is int
    assert lm == d.local_mem_size


def test_offload_array_mod():
    execution_queue = dpctl.SyclQueue()
    X = np.random.randint(low=1, high=2**16 - 1, size=10**6, dtype=np.int64)
    modulus_p = 347

    # Y is a regular NumPy array with NumPy allocated host memory
    Y = uqd.offloaded_array_mod(execution_queue, X, modulus_p)

    Ynp = X % modulus_p

    assert np.array_equal(Y, Ynp)
