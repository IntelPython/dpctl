.. _devices:

######
Device
######

A device is an abstract representation of an XPU. The :class:`dpctl.SyclDevice`
class represents a device and is a wrapper over the
:sycl_device:`sycl::device <>` SYCL* runtime class.

Creating Devices
----------------

The :class:`dpctl.SyclDevice` class includes the default constructor to create a
``default`` device. This device is selected by the SYCL* runtime. You can also use
explicit :ref:`filter selector strings <sec-filter-selection>` to create a
device.

.. note:: Refer to :ref:`device_selection` for more information. 

Listing Devices
---------------

:py:mod:`dpctl` provides the :func:`dpctl.get_devices` utility function to list
the available devices on a user's system. The list of devices returned depends
on the available hardware, installed drivers,
:dpcpp_envar:`environment variables <>` influencing SYCL* runtime,
such as ``SYCL_DEVICE_FILTER`` or ``SYCL_DEVICE_ALLOWLIST``.

.. _fig-listing-devices:

.. literalinclude:: ../../../../../examples/python/device_selection.py
    :language: python
    :lines: 20-22, 107-131
    :caption: Listing Available Devices
    :linenos:

A possible output for the :ref:`fig-listing-devices` example:

.. program-output:: python ../examples/python/device_selection.py -r list_devices

The :ref:`fig-listing-devices` example demonstrates the usage of
:func:`dpctl.get_devices`. 

You can filter the list based on the
:class:`dpctl.backend` and :class:`dpctl.device_type`. 

The 0-based ordinal position of a device in the output of :func:`dpctl.get_devices` corresponds to
the ``device id`` value in the filter selector string corresponding to the
device. For example, ``"opencl:cpu:0"`` refers to the first device in the list
returned by ``dpctl.get_devices(backend="opencl", device_type="cpu")``. If such
a list is empty, device construction call ``dpctl.SyclDevice("opencl:gpu:0")``
raises a ``ValueError``.

.. Note::

    Unless the system configuration changes, the list of devices returned by
    :func:`dpctl.get_devices` and the relative ordering of devices in the list
    is stable for every call to the function, even across different runs of an
    application.

Device Aspects and Information Descriptors
------------------------------------------

A device can have various *aspects* and *information descriptors* that describe
its hardware characteristics:

* :sycl_aspects:`Aspects <>` are boolean characteristics of the device
* :sycl_device_info:`information descriptors <>` are non-boolean characteristics
  that provide more verbose information about the device
* :class:`dpctl.SyclDevice` exposes various Python* properties that describe a
  device's aspects and information descriptors. 

For example, the property ``has_aspect_fp16`` returns a boolean expression indicating if:

* a particular device has the ``"fp16"`` aspect
* supports the IEEE-754 half-precision floating point type

The ``name`` property is
an information descriptor that returns a string with the name of the device.

.. _fig-available-properties:

.. code-block:: Python
    :caption: Listing Available Device Aspects and Information Descriptors
    :linenos:

    import dpctl
    import inspect

    def get_properties(cls, prop_name):
        "Get the name of properties of a class known to have `prop_name`"
        known_property_t = type(getattr(cls, prop_name))
        return [n for n, o in inspect.getmembers(cls) if isinstance(o, known_property_t)]

    print(len(get_properties(dpctl.SyclDevice, "name")))
    # Output: 52

The :ref:`fig-available-properties` example demonstrates a programmatic way to
list all the aspects and information descriptor properties in
:class:`dpctl.SyclDevice`.

.. _sec-devices-sub-devices:

Sub-devices
-----------

You can partition a device into sub-devices. 

A sub-device represents a subset of the computational units within a device 
that are grouped based on some hardware criteria. For example, you can partition a two-socket 
CPU into two sub-devices, where each sub-device represents a separate
:numa_domain:`NUMA domain <>`. Depending on the hardware characteristics and
the capabilities of the SYCL* runtime, a sub-device may be partitioned further.

For devices that support partitioning, you can use
:func:`dpctl.SyclDevice.create_sub_devices` to create a list of
sub-devices. The requested partitioning scheme is indicated with the usage of the
required ``partition`` keyword. 

Several types of partitioning schemes are available:

* **Count partitioning**
    The partitioning scheme is specified as a list of positive integers
    indicating a partitioning with each sub-device having the requested number
    of parallel compute units or as a single positive integer indicating
    equal-counts partition.

* **Affinity partitioning**
    The partitioning scheme is specified as a string indicating an affinity
    domain used to create sub-devices that share a common resource, such as
    certain hardware cache levels.

.. Note::

    Use ``partition="next_partitionable"`` to partition along the next level of
    architectural hierarchy.

The following example shows an affinity-based partitioning of a CPU device
into sub-devices based on the available NUMA* domains:

.. _fig-partition-cpu:

.. literalinclude:: ../../../../../examples/python/subdevices.py
    :language: python
    :lines: 17, 62-76
    :caption: Partitioning a CPU device
    :linenos:

A possible output for the :ref:`fig-partition-cpu` example:

.. program-output:: python ../examples/python/subdevices.py -r subdivide_by_affinity
