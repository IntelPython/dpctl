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

# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True

""" This module implements several device creation helper functions:

  - wrapper functions to create a SyclDevice from the standard SYCL
    device selector classes.
  - functions to return a list of devices based on a specified device_type or
    backend_type combination.
"""

from ._backend cimport (  # noqa: E211
    DPCTLAcceleratorSelector_Create,
    DPCTLCPUSelector_Create,
    DPCTLDefaultSelector_Create,
    DPCTLDevice_CreateFromSelector,
    DPCTLDeviceMgr_GetDevices,
    DPCTLDeviceMgr_GetNumDevices,
    DPCTLDeviceSelector_Delete,
    DPCTLDeviceVector_Delete,
    DPCTLDeviceVector_GetAt,
    DPCTLDeviceVector_Size,
    DPCTLDeviceVectorRef,
    DPCTLGPUSelector_Create,
    DPCTLHostSelector_Create,
    DPCTLSyclDeviceRef,
    DPCTLSyclDeviceSelectorRef,
    _backend_type,
    _device_type,
)

from .enum_types import backend_type
from .enum_types import device_type as device_type_t

__all__ = [
    "get_devices",
    "select_accelerator_device",
    "select_cpu_device",
    "select_default_device",
    "select_gpu_device",
    "select_host_device",
    "get_num_devices",
    "has_cpu_devices",
    "has_gpu_devices",
    "has_accelerator_devices",
    "has_host_device",
]


cdef _backend_type _string_to_dpctl_sycl_backend_ty(str backend_str):
    backend_str = backend_str.strip().lower()
    if backend_str == "all":
        return _backend_type._ALL_BACKENDS
    elif backend_str == "cuda":
        return _backend_type._CUDA
    elif backend_str == "host":
        return _backend_type._HOST
    elif backend_str == "level_zero":
        return _backend_type._LEVEL_ZERO
    elif backend_str == "opencl":
        return _backend_type._OPENCL
    else:
        return _backend_type._UNKNOWN_BACKEND


cdef _device_type _string_to_dpctl_sycl_device_ty(str dty_str):
    dty_str = dty_str.strip().lower()
    if dty_str == "accelerator":
        return _device_type._ACCELERATOR
    elif dty_str == "all":
        return _device_type._ALL_DEVICES
    elif dty_str == "automatic":
        return _device_type._AUTOMATIC
    elif dty_str == "cpu":
        return _device_type._CPU
    elif dty_str == "custom":
        return _device_type._CUSTOM
    elif dty_str == "gpu":
        return _device_type._GPU
    elif dty_str == "host_device":
        return _device_type._HOST_DEVICE
    else:
        return _device_type._UNKNOWN_DEVICE


cdef _backend_type _enum_to_dpctl_sycl_backend_ty(BTy):
    if BTy == backend_type.all:
        return _backend_type._ALL_BACKENDS
    elif BTy == backend_type.cuda:
        return _backend_type._CUDA
    elif BTy == backend_type.host:
        return _backend_type._HOST
    elif BTy == backend_type.level_zero:
        return _backend_type._LEVEL_ZERO
    elif BTy == backend_type.opencl:
        return _backend_type._OPENCL
    else:
        return _backend_type._UNKNOWN_BACKEND


cdef _device_type _enum_to_dpctl_sycl_device_ty(DTy):
    if DTy == device_type_t.all:
        return _device_type._ALL_DEVICES
    elif DTy == device_type_t.accelerator:
        return _device_type._ACCELERATOR
    elif DTy == device_type_t.automatic:
        return _device_type._AUTOMATIC
    elif DTy == device_type_t.cpu:
        return _device_type._CPU
    elif DTy == device_type_t.custom:
        return _device_type._CUSTOM
    elif DTy == device_type_t.gpu:
        return _device_type._GPU
    elif DTy == device_type_t.host_device:
        return _device_type._HOST_DEVICE
    else:
        return _device_type._UNKNOWN_DEVICE


cdef list _get_devices(DPCTLDeviceVectorRef DVRef):
    cdef list devices = []
    cdef size_t nelems = 0
    if DVRef:
        nelems = DPCTLDeviceVector_Size(DVRef)
        for i in range(0, nelems):
            DRef = DPCTLDeviceVector_GetAt(DVRef, i)
            D = SyclDevice._create(DRef)
            devices.append(D)

    return devices


cpdef list get_devices(backend=backend_type.all, device_type=device_type_t.all):
    """ Returns a list of :class:`dpctl.SyclDevice` instances selected based on
    the given :class:`dpctl.device_type` and :class:`dpctl.backend_type` values.

    The function is analogous to ``sycl::devices::get_devices()``, but with an
    additional functionality that allows filtering SYCL devices based on
    ``backend`` in addition to only ``device_type``.

    Args:
        backend (optional): Defaults to ``dpctl.backend_type.all``.
            A :class:`dpctl.backend_type` enum value or a string that
            specifies a SYCL backend. Currently, accepted values are: "cuda",
            "opencl", "level_zero", or "all".
        device_type (optional): Defaults to ``dpctl.device_type.all``.
            A :class:`dpctl.device_type` enum value or a string that
            specifies a SYCL device type. Currently, accepted values are:
            "gpu", "cpu", "accelerator", "host_device", or "all".
    Returns:
        list: A list of available :class:`dpctl.SyclDevice` instances that
        satisfy the provided :class:`dpctl.backend_type` and
        :class:`dpctl.device_type` values.
    """
    cdef _backend_type BTy = _backend_type._ALL_BACKENDS
    cdef _device_type DTy = _device_type._ALL_DEVICES
    cdef DPCTLDeviceVectorRef DVRef = NULL
    cdef list devices

    if isinstance(backend, str):
        BTy = _string_to_dpctl_sycl_backend_ty(backend)
    elif isinstance(backend, backend_type):
        BTy = _enum_to_dpctl_sycl_backend_ty(backend)
    else:
        raise TypeError(
            "backend should be specified as a str or an "
            "``enum_types.backend_type``."
        )

    if isinstance(device_type, str):
        DTy = _string_to_dpctl_sycl_device_ty(device_type)
    elif isinstance(device_type, device_type_t):
        DTy = _enum_to_dpctl_sycl_device_ty(device_type)
    else:
        raise TypeError(
            "device type should be specified as a str or an "
            "``enum_types.device_type``."
        )

    DVRef = DPCTLDeviceMgr_GetDevices(BTy | DTy)
    devices = _get_devices(DVRef)
    DPCTLDeviceVector_Delete(DVRef)

    return devices


cpdef int get_num_devices(
    backend=backend_type.all, device_type=device_type_t.all
):
    """ A helper function to return the number of SYCL devices of a given
    :class:`dpctl.device_type` and :class:`dpctl.backend_type`.

    Args:
        backend (optional): Defaults to ``dpctl.backend_type.all``.
            A :class:`dpctl.backend_type` enum value or a string that
            specifies a SYCL backend. Currently, accepted values are: "cuda",
            "opencl", "level_zero", or "all".
        device_type (optional): Defaults to ``dpctl.device_type.all``.
            A :class:`dpctl.device_type` enum value or a string that
            specifies a SYCL device type. Currently, accepted values are:
            "gpu", "cpu", "accelerator", "host_device", or "all".
    Returns:
        int: The number of available SYCL devices that satisfy the provided
        :class:`dpctl.backend_type` and :class:`dpctl.device_type` values.
    """
    cdef _backend_type BTy = _backend_type._ALL_BACKENDS
    cdef _device_type DTy = _device_type._ALL_DEVICES
    cdef int num_devices = 0

    if isinstance(backend, str):
        BTy = _string_to_dpctl_sycl_backend_ty(backend)
    elif isinstance(backend, backend_type):
        BTy = _enum_to_dpctl_sycl_backend_ty(backend)
    else:
        raise TypeError(
            "backend should be specified as a ``str`` or an "
            "``enum_types.backend_type``"
        )

    if isinstance(device_type, str):
        DTy = _string_to_dpctl_sycl_device_ty(device_type)
    elif isinstance(device_type, device_type_t):
        DTy = _enum_to_dpctl_sycl_device_ty(device_type)
    else:
        raise TypeError(
            "device type should be specified as a ``str`` or an "
            "``enum_types.device_type``"
        )

    num_devices = DPCTLDeviceMgr_GetNumDevices(BTy | DTy)

    return num_devices


cpdef cpp_bool has_cpu_devices():
    """ A helper function to check if there are any SYCL CPU devices available.

    Returns:
        bool: ``True`` if ``sycl::device_type::cpu`` devices are present,
        ``False`` otherwise.
    """
    cdef int num_cpu_dev = DPCTLDeviceMgr_GetNumDevices(_device_type._CPU)
    return <cpp_bool>num_cpu_dev


cpdef cpp_bool has_gpu_devices():
    """ A helper function to check if there are any SYCL GPU devices available.

    Returns:
        bool: ``True`` if ``sycl::device_type::gpu`` devices are present,
        ``False`` otherwise.
    """
    cdef int num_gpu_dev = DPCTLDeviceMgr_GetNumDevices(_device_type._GPU)
    return <cpp_bool>num_gpu_dev


cpdef cpp_bool has_accelerator_devices():
    """ A helper function to check if there are any SYCL Accelerator devices
    available.

    Returns:
        bool: ``True`` if ``sycl::device_type::accelerator`` devices are
        present, ``False`` otherwise.
    """
    cdef int num_accelerator_dev = DPCTLDeviceMgr_GetNumDevices(
        _device_type._ACCELERATOR
    )
    return <cpp_bool>num_accelerator_dev


cpdef cpp_bool has_host_device():
    """ A helper function to check if there are any SYCL Host devices available.

    Returns:
        bool: ``True`` if ``sycl::device_type::host`` devices are present,
        ``False`` otherwise.
    """
    cdef int num_host_dev = DPCTLDeviceMgr_GetNumDevices(
        _device_type._HOST_DEVICE
    )
    return <cpp_bool>num_host_dev


cpdef SyclDevice select_accelerator_device():
    """ A wrapper for SYCL's ``accelerator_selector`` class.

    Returns:
        dpctl.SyclDevice: A Python object wrapping the SYCL ``device``
        returned by the SYCL ``accelerator_selector``.
    Raises:
        ValueError: If the SYCL ``accelerator_selector`` is unable to select a
                    ``device``.
    """
    cdef DPCTLSyclDeviceSelectorRef DSRef = DPCTLAcceleratorSelector_Create()
    cdef DPCTLSyclDeviceRef DRef = DPCTLDevice_CreateFromSelector(DSRef)
    # Free up the device selector
    DPCTLDeviceSelector_Delete(DSRef)
    if DRef is NULL:
        raise ValueError("Device unavailable.")
    Device = SyclDevice._create(DRef)
    return Device


cpdef SyclDevice select_cpu_device():
    """ A wrapper for SYCL's ``cpu_selector`` class.

    Returns:
        dpctl.SyclDevice: A Python object wrapping the SYCL ``device``
        returned by the SYCL ``cpu_selector``.
    Raises:
        ValueError: If the SYCL ``cpu_selector`` is unable to select a
                    ``device``.
    """
    cdef DPCTLSyclDeviceSelectorRef DSRef = DPCTLCPUSelector_Create()
    cdef DPCTLSyclDeviceRef DRef = DPCTLDevice_CreateFromSelector(DSRef)
    # Free up the device selector
    DPCTLDeviceSelector_Delete(DSRef)
    if DRef is NULL:
        raise ValueError("Device unavailable.")
    Device = SyclDevice._create(DRef)
    return Device


cpdef SyclDevice select_default_device():
    """ A wrapper for SYCL's ``default_selector`` class.

    Returns:
        dpctl.SyclDevice: A Python object wrapping the SYCL ``device``
        returned by the SYCL ``default_selector``.
    Raises:
        ValueError: If the SYCL ``default_selector`` is unable to select a
            ``device``.
    """
    cdef DPCTLSyclDeviceSelectorRef DSRef = DPCTLDefaultSelector_Create()
    cdef DPCTLSyclDeviceRef DRef = DPCTLDevice_CreateFromSelector(DSRef)
    # Free up the device selector
    DPCTLDeviceSelector_Delete(DSRef)
    if DRef is NULL:
        raise ValueError("Device unavailable.")
    Device = SyclDevice._create(DRef)
    return Device


cpdef SyclDevice select_gpu_device():
    """ A wrapper for SYCL's ``gpu_selector`` class.

    Returns:
        dpctl.SyclDevice: A Python object wrapping the SYCL ``device``
        returned by the SYCL ``gpu_selector``.
    Raises:
        ValueError: If the SYCL ``gpu_selector`` is unable to select a
                    ``device``.
    """
    cdef DPCTLSyclDeviceSelectorRef DSRef = DPCTLGPUSelector_Create()
    cdef DPCTLSyclDeviceRef DRef = DPCTLDevice_CreateFromSelector(DSRef)
    # Free up the device selector
    DPCTLDeviceSelector_Delete(DSRef)
    if DRef is NULL:
        raise ValueError("Device unavailable.")
    Device = SyclDevice._create(DRef)
    return Device


cpdef SyclDevice select_host_device():
    """ A wrapper for SYCL's ``host_selector`` class.

    Returns:
        dpctl.SyclDevice: A Python object wrapping the SYCL ``device``
        returned by the SYCL ``host_selector``.
    Raises:
        ValueError: If the SYCL ``host_selector`` is unable to select a
            ``device``.
    """
    cdef DPCTLSyclDeviceSelectorRef DSRef = DPCTLHostSelector_Create()
    cdef DPCTLSyclDeviceRef DRef = DPCTLDevice_CreateFromSelector(DSRef)
    # Free up the device selector
    DPCTLDeviceSelector_Delete(DSRef)
    if DRef is NULL:
        raise ValueError("Device unavailable.")
    Device = SyclDevice._create(DRef)
    return Device
