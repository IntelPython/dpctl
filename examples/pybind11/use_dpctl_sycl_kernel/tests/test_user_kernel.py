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

import os.path

import numpy as np
import pytest
import use_kernel as uk

import dpctl
import dpctl.memory as dpmem
import dpctl.program as dppr


def _get_spv_path():
    uk_dir = os.path.dirname(os.path.abspath(uk.__file__))
    proj_dir = os.path.dirname(uk_dir)
    return os.path.join(proj_dir, "resource", "double_it.spv")


def test_spv_file_exists():
    assert os.path.exists(_get_spv_path())


def test_kernel_can_be_found():
    fn = _get_spv_path()
    with open(fn, "br") as f:
        il = f.read()
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Could not create default queue")
    pr = dppr.create_program_from_spirv(q, il, "")
    assert pr.has_sycl_kernel("double_it")


def test_kernel_submit_through_extension():
    fn = _get_spv_path()
    with open(fn, "br") as f:
        il = f.read()
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Could not create default queue")
    pr = dppr.create_program_from_spirv(q, il, "")
    krn = pr.get_sycl_kernel("double_it")
    assert krn.num_args == 2

    x = np.arange(0, stop=13, step=1, dtype="i4")
    y = np.empty_like(x)

    x_usm = dpmem.MemoryUSMDevice(x.nbytes, queue=q)
    y_usm = dpmem.MemoryUSMDevice(y.nbytes, queue=q)

    ev = q.memcpy_async(dest=x_usm, src=x, count=x_usm.nbytes)

    uk.submit_custom_kernel(q, krn, x_usm, y_usm, [ev])

    q.memcpy(dest=y, src=y_usm, count=y.nbytes)

    assert np.array_equal(y, np.arange(0, 26, step=2, dtype="i4"))
