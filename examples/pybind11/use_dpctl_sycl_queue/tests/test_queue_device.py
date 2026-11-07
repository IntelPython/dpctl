#                      Data Parallel Control (dpctl)
#
# Copyright 2022 Intel Corporation
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
import pytest
import use_queue_device as uqd

import dpctl

skip_no_bfloat16 = pytest.mark.skipif(
    not uqd.has_bfloat16,
    reason="The DPC++ compiler the extension was built with does not provide "
    "the bfloat16 extension",
)


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


def test_get_sub_group_sizes():
    d = dpctl.SyclDevice()
    szs = uqd.get_sub_group_sizes(d)
    assert type(szs) is list
    assert all(type(el) is int for el in szs)
    szs == d.sub_group_sizes


def test_get_partition_max_sub_devices():
    d = dpctl.SyclDevice()
    mt = uqd.get_partition_max_sub_devices(d)
    assert type(mt) is int
    assert mt >= 0
    assert mt <= d.max_compute_units


@skip_no_bfloat16
def test_bfloat16_round_trip():
    # a Python float is rounded to the nearest bfloat16, and comes back as the
    # float that bfloat16 stands for
    assert uqd.bfloat16_round_trip(1.5) == 1.5
    assert uqd.bfloat16_round_trip(3.14) == 3.140625
    assert type(uqd.bfloat16_round_trip(3.14)) is float
    # an integer converts, too
    assert uqd.bfloat16_round_trip(3) == 3.0


@skip_no_bfloat16
def test_bfloat16_round_trip_ml_dtypes_scalar():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    value = ml_dtypes.bfloat16(3.14)
    assert uqd.bfloat16_round_trip(value) == float(value)


@skip_no_bfloat16
def test_offloaded_bfloat16_scale():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    execution_queue = dpctl.SyclQueue()
    X = np.arange(1, 11, dtype=np.float32)
    factor = ml_dtypes.bfloat16(1.1)

    Y = uqd.offloaded_bfloat16_scale(execution_queue, X, factor)

    assert np.array_equal(Y, X * np.float32(factor))
