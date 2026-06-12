#                      Data Parallel Control (dpctl)
#
# Copyright 2020 Intel Corporation
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

"""Defines Python enumeration types for SYCL enumerations.

This module provides two enumeration types corresponding to SYCL's
backend and device_type enumerations.

"""

from enum import Enum, auto

__all__ = [
    "device_type",
    "backend_type",
    "event_status_type",
    "global_mem_cache_type",
    "local_mem_type",
    "partition_property",
    "fp_config",
    "memory_order",
    "memory_scope",
]


class device_type(Enum):
    """
    An :class:`enum.Enum` of supported SYCL device types.

        |  ``all``
        |  ``accelerator``
        |  ``automatic``
        |  ``cpu``
        |  ``custom``
        |  ``gpu``

    :Example:
        .. code-block:: python

            import dpctl

            # filter GPU devices amongst available SYCL devices
            gpu_devs = [
                d for d in dpctl.get_devices() if (
                    d.device_type == dpctl.device_type.gpu
                ) ]

            # alternatively, get GPU devices directly
            gpu_devs2 = dpctl.get_devices(device_type=dpctl.device_type.gpu)
    """

    all = auto()
    accelerator = auto()
    automatic = auto()
    cpu = auto()
    custom = auto()
    gpu = auto()


class backend_type(Enum):
    """
    An :class:`enum.Enum` of supported SYCL backends.

        |  ``all``
        |  ``cuda``
        |  ``hip``
        |  ``level_zero``
        |  ``opencl``

    :Example:
        .. code-block:: python

            import dpctl

            # create a SYCL device with OpenCL backend using filter selector
            d = dpctl.SyclDevice("opencl")
            d.backend
            # Possible output: <backend_type.opencl: 5>
    """

    all = auto()
    cuda = auto()
    hip = auto()
    level_zero = auto()
    opencl = auto()


class event_status_type(Enum):
    """
    An :class:`enum.Enum` of SYCL event states.

        |  ``unknown_status``
        |  ``submitted``
        |  ``running``
        |  ``complete``

    :Example:
        .. code-block:: python

            import dpctl
            ev = dpctl.SyclEvent()
            ev.execution_status
            # Possible output: <event_status_type.complete: 4>
    """

    unknown_status = auto()
    submitted = auto()
    running = auto()
    complete = auto()


class global_mem_cache_type(Enum):
    """
    An :class:`enum.Enum` of global memory cache types for a device.

        |  ``indeterminate``
        |  ``none``
        |  ``read_only``
        |  ``read_write``

    :Example:
        .. code-block:: python

            import dpctl
            dev = dpctl.SyclDevice()
            dev.global_mem_cache_type
            # Possible output: <global_mem_cache_type.read_write: 4>
    """

    indeterminate = auto()
    none = auto()
    read_only = auto()
    read_write = auto()


class local_mem_type(Enum):
    """
    An :class:`enum.Enum` of local memory types for a device.

        |  ``none``
        |  ``local``
        |  ``global_mem``

    :Example:
        .. code-block:: python

            import dpctl
            dev = dpctl.SyclDevice()
            dev.local_mem_type
            # Possible output: <local_mem_type.local: 2>
    """

    none = auto()
    local = auto()
    global_mem = auto()


class partition_property(Enum):
    """
    An :class:`enum.Enum` of partition property types.

        |  ``no_partition``
        |  ``partition_equally``
        |  ``partition_by_counts``
        |  ``partition_by_affinity_domain``

    :Example:
        .. code-block:: python

            import dpctl
            dev = dpctl.SyclDevice()
            dev.partition_type_property
            # Possible output: <partition_property.no_partition: 1>
    """

    no_partition = auto()
    partition_equally = auto()
    partition_by_counts = auto()
    partition_by_affinity_domain = auto()


class fp_config(Enum):
    """
    An :class:`enum.Enum` of floating-point capability flags.

        |  ``denorm``
        |  ``inf_nan``
        |  ``round_to_nearest``
        |  ``round_to_zero``
        |  ``round_to_inf``
        |  ``fma``
        |  ``correctly_rounded_divide_sqrt``
        |  ``soft_float``

    :Example:
        .. code-block:: python

            import dpctl
            dev = dpctl.SyclDevice()
            dev.single_fp_config
            # Possible output: (
            #     <fp_config.denorm: 1>,
            #     <fp_config.inf_nan: 2>,
            #     ...
            # )
    """

    denorm = auto()
    inf_nan = auto()
    round_to_nearest = auto()
    round_to_zero = auto()
    round_to_inf = auto()
    fma = auto()
    correctly_rounded_divide_sqrt = auto()
    soft_float = auto()


class memory_order(Enum):
    """
    An :class:`enum.Enum` of memory ordering capabilities.

        |  ``relaxed``
        |  ``acquire``
        |  ``release``
        |  ``acq_rel``
        |  ``seq_cst``

    :Example:
        .. code-block:: python

            import dpctl
            dev = dpctl.SyclDevice()
            dev.atomic_memory_order_capabilities
            # Possible output: (
            #     <memory_order.relaxed: 1>,
            #     <memory_order.acquire: 2>,
            #     ...
            # )
    """

    relaxed = auto()
    acquire = auto()
    release = auto()
    acq_rel = auto()
    seq_cst = auto()


class memory_scope(Enum):
    """
    An :class:`enum.Enum` of memory scope capabilities.

        |  ``work_item``
        |  ``sub_group``
        |  ``work_group``
        |  ``device``
        |  ``system``

    :Example:
        .. code-block:: python

            import dpctl
            dev = dpctl.SyclDevice()
            dev.atomic_memory_scope_capabilities
            # Possible output: (
            #     <memory_scope.work_item: 1>,
            #     <memory_scope.sub_group: 2>,
            #     ...
            # )
    """

    work_item = auto()
    sub_group = auto()
    work_group = auto()
    device = auto()
    system = auto()
