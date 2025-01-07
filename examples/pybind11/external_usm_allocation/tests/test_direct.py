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

import dpctl
import dpctl.memory as dpm
import dpctl.tensor as dpt


def test_direct():
    q = dpctl.SyclQueue()

    nb = 2 * 30
    mbuf = eua.make_zeroed_device_memory(nb, q)

    assert isinstance(mbuf, dpm.MemoryUSMDevice)
    assert mbuf.nbytes == 2 * 30
    assert mbuf.sycl_queue == q

    x = dpt.usm_ndarray(30, dtype="i2", buffer=mbuf)
    assert dpt.all(x == dpt.zeros(30, dtype="i2", sycl_queue=q))
