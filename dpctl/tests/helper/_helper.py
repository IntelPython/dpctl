#                      Data Parallel Control (dpctl)
#
# Copyright 2021 Intel Corporation
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


def get_queue_or_skip(args=()):
    try:
        q = dpctl.SyclQueue(*args)
    except dpctl.SyclQueueCreationError:
        pytest.skip(f"Queue could not be created from {args}")
    return q


def make_kernel_of_unknown_origin(krn):
    """Wraps the ``sycl::kernel`` of ``krn`` with the ``SyclKernel_Make``
    C-API function, as a kernel created outside of ``dpctl.compiler`` is.
    """
    import ctypes
    import sys

    mod = sys.modules[krn.__class__.__module__]
    cap_ptr_fn = ctypes.pythonapi.PyCapsule_GetPointer
    cap_ptr_fn.restype = ctypes.c_void_p
    cap_ptr_fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
    make_fn_ptr = cap_ptr_fn(
        mod.__pyx_capi__["SyclKernel_Make"],
        b"struct PySyclKernelObject *(DPCTLSyclKernelRef, char const *)",
    )
    make_fn = ctypes.PYFUNCTYPE(
        ctypes.py_object, ctypes.c_void_p, ctypes.c_char_p
    )(make_fn_ptr)
    return make_fn(krn.addressof_ref(), krn.get_function_name().encode())
