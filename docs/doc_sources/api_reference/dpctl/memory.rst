.. _dpctl_memory_pyapi:


:py:mod:`dpctl.memory`
======================

Subpackage :py:mod:`dpctl.memory` exposes Unified Shared Memory(USM) operations.

Unified Shared Memory is a pointer-based memory management in SYCL guaranteeing that
the host and all devices use a `unified address space <sycl_unified_address_space_>`_.
Quoting from the SYCL specification:

.. _sycl_unified_address_space: https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#_unified_addressing

    Pointer values in the unified address space will always refer to the same location in memory.
    The unified address space encompasses the host and one or more devices. Note that this does
    not require addresses in the unified address space to be accessible on all devices, just that
    pointer values will be consistent.

Three types of USM allocations are supported:

.. list-table::
    :widths: 10 90
    :header-rows: 1

    * - USM allocation type
      - Description
    * - ``"device"``
      - Allocations in device memory accessible by the device but **not** by the host
    * - ``"shared"``
      - Allocations in device memory accessible by both the host and the device
    * - ``"host"``
      - Allocations in host memory accessiblle by both the host and the device


.. py:module:: dpctl.memory


.. currentmodule:: dpctl.memory

.. rubric:: Python classes representing USM allocations

.. autosummary::
    :toctree: generated
    :template: autosummary/usmmemory.rst
    :nosignatures:

    MemoryUSMDevice
    MemoryUSMShared
    MemoryUSMHost

Python objects representing USM allocations provide ``__sycl_usm_array_interface__``  :ref:`attribute <suai_attribute>`.
A Python object can be converted to one of these classes using the following function:

.. autosummary::
    :toctree: generated
    :nosignatures:

    as_usm_memory

Should the USM allocation fail, the following Python exception will be raised:

.. autosummary::
    :toctree: generated
    :nosignatures:

    USMAllocationError

.. toctree::
    :hidden:

    sycl_usm_array_interface
