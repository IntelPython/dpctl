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

import pytest

import dpctl


def has_gpu(backend="opencl"):
    return bool(dpctl.get_num_devices(backend=backend, device_type="gpu"))


def has_cpu(backend="opencl"):
    return bool(dpctl.get_num_devices(backend=backend, device_type="cpu"))


def has_sycl_platforms():
    return bool(len(dpctl.get_platforms()))


def create_invalid_capsule():
    """Creates an invalid capsule for the purpose of testing dpctl
    constructors that accept capsules.
    """
    import ctypes

    ctor = ctypes.pythonapi.PyCapsule_New
    ctor.restype = ctypes.py_object
    ctor.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    return ctor(id(ctor), b"invalid", 0)


def get_queue_or_skip(args=tuple()):
    try:
        q = dpctl.SyclQueue(*args)
    except dpctl.SyclQueueCreationError:
        pytest.skip(f"Queue could not be created from {args}")
    return q


def skip_if_dtype_not_supported(dt, q_or_dev):
    import dpctl.tensor as dpt

    dt = dpt.dtype(dt)
    if type(q_or_dev) is dpctl.SyclQueue:
        dev = q_or_dev.sycl_device
    elif type(q_or_dev) is dpctl.SyclDevice:
        dev = q_or_dev
    else:
        raise TypeError(
            "Expected dpctl.SyclQueue or dpctl.SyclDevice, "
            f"got {type(q_or_dev)}"
        )
    dev_has_dp = dev.has_aspect_fp64
    if dev_has_dp is False and dt in [dpt.float64, dpt.complex128]:
        pytest.skip(
            f"{dev.name} does not support double precision floating point types"
        )
    dev_has_hp = dev.has_aspect_fp16
    if dev_has_hp is False and dt in [
        dpt.float16,
    ]:
        pytest.skip(
            f"{dev.name} does not support half precision floating point type"
        )
