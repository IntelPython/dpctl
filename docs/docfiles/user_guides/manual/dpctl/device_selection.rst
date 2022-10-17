.. _device_selection:

################
Device Selection
################

Device selection refers to programmatically selecting a single device from
the set of :ref:`devices <devices>` available on the system.

Selecting a Specific Type of Device
-----------------------------------

If you need to select a specific type of device, such as a GPU,
use one of the helper functions included inside `dpctl`` directly. Dpctl includes
:ref:`helper functions <dpctl_device_selection_functions>` for selecting:

* ``host``
* ``cpu``
* ``gpu``
* ``accelerator``
*  ``default`` device


These functions are analogous to SYCL* built-in
:sycl_device_selector:`sycl::device_selector <>` classes. The scoring and
selection of a specific device when multiple devices of the same type are
available on a system are deferred to the underlying SYCL* runtime.

The example :ref:`fig-gpu-device-selection` shows the usage of the
:func:`dpctl.select_gpu_device()` device selection function. In case when multiple
GPU devices are available, only one is returned based on the underlying scoring
logic inside of the SYCL* runtime. If the selection function is unable to select a
device, a ``ValueError`` is raised.

.. _fig-gpu-device-selection:

.. literalinclude:: ../../../../../examples/python/device_selection.py
    :language: python
    :lines: 20-21, 38-52
    :caption: Selecting a GPU Device
    :linenos:

A possible output for the :ref:`fig-gpu-device-selection` example:

.. program-output:: python ../examples/python/device_selection.py -r create_gpu_device

.. _sec-filter-selection:

Selecting a Device Using a Filter String
----------------------------------------

Along with using the default device selection functions, a more explicit way of
device selection involves the usage of *filter strings*. Refer to
:oneapi_filter_selection:`oneAPI filter selection extension <>` to learn more.

The :ref:`fig-gpu-device-selection` example also demonstrates the usage of a filter string
to create a GPU device directly. Using a filter string allows much more
fine-grained control for selecting a device. 

The following :ref:`fig-filter-selection`
example demonstrates the usage of the device selection using filter
strings.

.. _fig-filter-selection:

.. literalinclude:: ../../../../../examples/python/filter_selection.py
    :language: python
    :lines: 20-21, 23-53
    :caption: Device Creation With Filter Strings
    :linenos:

A possible output for the :ref:`fig-filter-selection` example:

.. program-output:: python ../examples/python/filter_selection.py -r select_using_filter


It is also possible to pass a list of devices using a filter string. The
:ref:`fig-adv-device-selection` example demonstrates this use case. The
filter string ``gpu,cpu`` implies that a GPU should be selected if available,
otherwise a CPU device should be selected.

.. _fig-adv-device-selection:

.. literalinclude:: ../../../../../examples/python/device_selection.py
    :language: python
    :lines: 20-21, 55-67
    :caption: Selecting a GPU Device if Available
    :linenos:

A possible output for the :ref:`fig-adv-device-selection` example:

.. program-output:: python ../examples/python/device_selection.py -r create_gpu_device_if_present


A **filter string** is a three-tuple that may specify the *backend*,
*device type*, and *device number* as a colon (:) separated string. 


.. list-table:: Title
   :widths: 50 50
   :header-rows: 1

   * - String
     - Usage
     - Value
   * - *backend*
     - *device type*
     - *device number*
   * - Specifies the type of device driver.
     - Specifies the type of device.
     - Specifies the ordinality of the device in the listing of devices as
       determined by the SYCL* runtime.
   * - ``host``, ``opencl``, ``level-zero``, ``cuda``
     - ``host``, ``gpu``, ``cpu``, ``accelerator``
     - Numeric value
   
The backend, device type, and device number value are optional but provide at least one of them. 
That is, ``opencl:gpu:0``, ``gpu:0``, ``gpu``, ``0``, and ``opencl:0`` are all valid filter strings.

The device listing including the ``device number value`` remains stable for
a given system unless the driver configuration is changed or the SYCL*
runtime setting is changed using the ``SYCL_DEVICE_FILTER`` environment variable. 
Refer to :oneapi_filter_selection:`oneAPI filter selection extension <>` for more
information.

Advanced Device Selection
-------------------------

Real-world applications may require more precise control over device selection. 
Dpctl helps you to accomplish more advanced device selection.

.. _fig-custom-device-selection:

.. literalinclude:: ../../../../../examples/python/device_selection.py
    :language: python
    :lines: 20-21, 70-91
    :caption: Custom Device Selection
    :linenos:

The :ref:`fig-custom-device-selection` example shows a way of selecting a device
based on a specific hardware property. The process is the following:

1. The :func:`dpctl.get_devices()` returns a list of all *root* devices on the system. 
2. Out of that list the devices that
support half-precision floating-point arithmetic are selected. 
3. A "score" computed using the SYCL8 runtime's default device scoring logic that is
stored in :attr:`dpctl.SyclDevice.default_selector_score` is used to select a
single device. 

Refer to the :class:`dpctl.SyclDevice` documentation for a list
of hardware properties that may be used for device selection.

.. _RootDevice:

.. Note::
    A **root** device implies an unpartitioned device. A root device can be
    partitioned into two or more :ref:`sub-devices <sec-devices-sub-devices>`
    based on various criteria. For example, a CPU device with multiple NUMA*
    domains may be partitioned into multiple sub-devices, each representing a
    sub-device.

A convenience function :func:`dpctl.select_device_with_aspects()` is available,
which makes it easy to select a device based on a set of specific aspects. The
:ref:`fig-select-device-with-aspects` example selects a device that
supports double precision arithmetic and SYCL* USM shared memory allocation.

.. _fig-select-device-with-aspects:

.. literalinclude:: ../../../../../examples/python/device_selection.py
    :language: python
    :lines: 20-21, 94-103
    :caption: Device Selection Using Aspects
    :linenos:

A possible output for the :ref:`fig-select-device-with-aspects` example:

.. program-output:: python ../examples/python/device_selection.py -r create_device_with_aspects
