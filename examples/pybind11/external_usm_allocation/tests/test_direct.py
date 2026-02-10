#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2025 Intel Corporation
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

import external_usm_allocation as eua
import numpy as np

import dpctl
import dpctl.memory as dpm


def test_direct():
    q = dpctl.SyclQueue()

    nb = 2 * 30
    mbuf = eua.make_zeroed_device_memory(nb, q)

    assert isinstance(mbuf, dpm.MemoryUSMDevice)
    assert mbuf.nbytes == nb
    assert mbuf.sycl_queue == q

    x = np.empty(30, dtype="i2")
    assert x.nbytes == nb

    q.memcpy(dest=x, src=mbuf, count=nb)
    assert np.all(x == 0)
