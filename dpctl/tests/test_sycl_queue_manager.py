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

"""Defines unit test cases for the SyclQueueManager class.
"""

import pytest

import dpctl

from ._helper import has_cpu, has_gpu, has_sycl_platforms


@pytest.mark.skipif(
    not has_sycl_platforms(), reason="No SYCL platforms available"
)
def test_is_in_device_context_outside_device_ctxt():
    assert not dpctl.is_in_device_context()


@pytest.mark.skipif(not has_gpu(), reason="No OpenCL GPU queues available")
def test_is_in_device_context_inside_device_ctxt_gpu():
    with dpctl.device_context("opencl:gpu:0"):
        assert dpctl.is_in_device_context()


@pytest.mark.skipif(not has_cpu(), reason="No OpenCL CPU queues available")
def test_is_in_device_context_inside_device_ctxt_cpu():
    with dpctl.device_context("opencl:cpu:0"):
        assert dpctl.is_in_device_context()


@pytest.mark.skipif(not has_gpu(), reason="No OpenCL GPU queues available")
@pytest.mark.skipif(not has_cpu(), reason="No OpenCL CPU queues available")
def test_is_in_device_context_inside_nested_device_ctxt():
    with dpctl.device_context("opencl:cpu:0"):
        with dpctl.device_context("opencl:gpu:0"):
            assert dpctl.is_in_device_context()
        assert dpctl.is_in_device_context()
    assert not dpctl.is_in_device_context()


@pytest.mark.skipif(not has_cpu(), reason="No OpenCL CPU queues available")
def test_is_in_device_context_inside_nested_device_ctxt_cpu():
    cpu = dpctl.SyclDevice("cpu")
    n = cpu.max_compute_units
    n_half = n // 2
    try:
        d0, d1 = cpu.create_subdevices(partition=[n_half, n - n_half])
    except Exception:
        pytest.skip("Could not create subdevices")
    assert 0 == dpctl.get_num_activated_queues()
    with dpctl.device_context(d0):
        assert 1 == dpctl.get_num_activated_queues()
        with dpctl.device_context(d1):
            assert 2 == dpctl.get_num_activated_queues()
            assert dpctl.is_in_device_context()
        assert dpctl.is_in_device_context()
        assert 1 == dpctl.get_num_activated_queues()
    assert not dpctl.is_in_device_context()
    assert 0 == dpctl.get_num_activated_queues()


@pytest.mark.skipif(
    not has_sycl_platforms(), reason="No SYCL platforms available"
)
def test_get_current_device_type_outside_device_ctxt():
    assert dpctl.get_current_device_type() is not None


@pytest.mark.skipif(
    not has_sycl_platforms(), reason="No SYCL platforms available"
)
@pytest.mark.skipif(not has_gpu(), reason="No OpenCL GPU queues available")
def test_get_current_device_type_inside_device_ctxt():
    assert dpctl.get_current_device_type() is not None

    with dpctl.device_context("opencl:gpu:0"):
        assert dpctl.get_current_device_type() == dpctl.device_type.gpu

    assert dpctl.get_current_device_type() is not None


@pytest.mark.skipif(not has_cpu(), reason="No OpenCL CPU queues available")
@pytest.mark.skipif(not has_gpu(), reason="No OpenCL GPU queues available")
def test_get_current_device_type_inside_nested_device_ctxt():
    assert dpctl.get_current_device_type() is not None

    with dpctl.device_context("opencl:cpu:0"):
        assert dpctl.get_current_device_type() == dpctl.device_type.cpu

        with dpctl.device_context("opencl:gpu:0"):
            assert dpctl.get_current_device_type() == dpctl.device_type.gpu
        assert dpctl.get_current_device_type() == dpctl.device_type.cpu

    assert dpctl.get_current_device_type() is not None


@pytest.mark.skipif(
    not has_sycl_platforms(), reason="No SYCL platforms available"
)
def test_num_current_queues_outside_with_clause():
    assert 0 == dpctl.get_num_activated_queues()


@pytest.mark.skipif(not has_gpu(), reason="No OpenCL GPU queues available")
@pytest.mark.skipif(not has_cpu(), reason="No OpenCL CPU queues available")
def test_num_current_queues_inside_with_clause():
    with dpctl.device_context("opencl:cpu:0"):
        assert 1 == dpctl.get_num_activated_queues()
        with dpctl.device_context("opencl:gpu:0"):
            assert 2 == dpctl.get_num_activated_queues()
    assert 0 == dpctl.get_num_activated_queues()


@pytest.mark.skipif(not has_gpu(), reason="No OpenCL GPU queues available")
@pytest.mark.skipif(not has_cpu(), reason="No OpenCL CPU queues available")
def test_num_current_queues_inside_threads():
    from threading import Thread

    def SessionThread():
        assert dpctl.get_num_activated_queues() == 0
        with dpctl.device_context("opencl:gpu:0"):
            assert dpctl.get_num_activated_queues() == 1

    Session1 = Thread(target=SessionThread())
    Session2 = Thread(target=SessionThread())
    with dpctl.device_context("opencl:cpu:0"):
        assert dpctl.get_num_activated_queues() == 1
        Session1.start()
        Session2.start()


@pytest.mark.skipif(
    not has_sycl_platforms(), reason="No SYCL platforms available"
)
def test_get_current_backend():
    dpctl.get_current_backend()
    dpctl.get_current_device_type()
    q = dpctl.SyclQueue()
    dpctl.set_global_queue(q)
    if has_gpu():
        dpctl.set_global_queue("gpu")
    elif has_cpu():
        dpctl.set_global_queue("cpu")
