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
