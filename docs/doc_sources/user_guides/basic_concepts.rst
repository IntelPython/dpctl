.. _basic_concepts:

Heterogeneous Systems and Programming Concepts
==============================================

This section introduces the basic concepts defined by SYCL standard
for programming heterogeneous system, and used by :py:mod:`dpctl`.

.. note::
   For SYCL-level details, refer to a more topical SYCL reference,
   such as the :sycl_spec_2020:`SYCL 2020 spec <>`.

Definitions
-----------

* **Heterogeneous computing**
   Refers to computing on multiple devices in a program.

* **Host**
   Every program starts by running on a host, and most of the lines of code in
   a program, in particular lines of code implementing the Python interpreter
   itself, are usually for the host. Hosts are customarily CPUs.

* **Device**
   A device is a processing unit connected to a host that is programmable
   with a specific device driver. Different types of devices can have
   different architectures (CPUs, GPUs, FPGA, ASICs, DSP) but are programmable
   using the same :oneapi:`oneAPI <>` programming model.

* **Platform**
   Platform is an abstraction to represent a collection of devices addressable
   by the same lower-level framework. As multiple
   devices of the same type can programmed by the same framework, a platform may
   contain multiple devices. The same physical hardware (for example, GPU)
   may be programmable by different lower-level frameworks, and hence be enumerated
   as part of different platforms. For example, the same GPU hardware can be listed
   as an OpenCL* GPU device and a Level-Zero* GPU device.

* **Context**
   Holds the runtime information needed to operate on a device or a
   group of devices from the same platform. Contexts are relatively expensive
   to create and should be reused as much as possible.

* **Queue**
   A queue is needed to schedule the execution of any computation or data
   copying on the device. Queue construction requires specifying a device
   and a context targeting that device as well as additional properties,
   such as whether profiling information should be collected or submitted
   tasks are executed in the order in which they were submitted.

* **Event**
   An event holds information related to computation/data movement operation
   scheduled for execution on a queue, such as its execution status as well
   as profiling information if the queue the task was submitted to allowed
   for collection of such information. Events can be used to specify task
   dependencies as well as to synchronize host and devices.

* **Unified Shared Memory**
   Unified Shared Memory (USM) refers to pointer-based device memory management.
   USM allocations are bound to context. It means, a pointer representing
   USM allocation can be unambiguously mapped to the data it represents *only
   if* the associated context is known. USM allocations are accessible by
   computational kernels that are executed on a device, provided that the
   allocation is bound to the same context that is used to construct the queue
   where the kernel is scheduled for execution.

   Depending on the capability of the device, USM allocations can be:

.. csv-table::
   :header: "Name", "Host accessible", "Device accessibility"
   :widths: 25, 25, 50

   "Device allocation", "No","Refers to an allocation in host memory that is accessible from a device."
   "Shared allocation", "Yes", "Accessible by both the host and device."
   "Host allocation", "Yes", "Accessible by both the host and device."

Runtime manages synchronization of the host's and device's view into shared allocations.
The initial placement of the shared allocations is not defined.

* **Backend**
   Refers to the implementation of :oneapi:`oneAPI <>` programming model using a
   lower-level heterogeneous programming API. Amongst examples of backends are
   "cuda", "hip", "level_zero", "opencl". In particular backend implements a
   platform abstraction.


Platform
--------

A platform abstracts one or more SYCL devices that are connected to
a host and can be programmed by the same underlying framework.

The :class:`dpctl.SyclPlatform` class represents a platform and
abstracts the :sycl_platform:`sycl::platform <>` SYCL runtime class.

To obtain all platforms available on a system programmatically, use
:func:`dpctl.lsplatform` function. Refer to :ref:`Enumerating available devices <beginners_guide_enumerating_devices>`
for more information.

It is possible to select devices from specific backend, and hence belonging to
the same platform, by :ref:`using <beginners_guide_oneapi_device_selector>`
``ONEAPI_DEVICE_SELECTOR`` environment variable, or by using
a :ref:`filter selector string <filter_selector_string>`.


Context
-------

A context is an entity that is associated with the state of device as managed by the
backend. The context is required to map unified address space pointer to the device
where it was allocated unambiguously.

In order for two DPC++-based Python extensions to share USM allocations, e.g.
as part of :ref:`DLPack exchange <dpctl_tensor_dlpack_support>`, they each must use
the `same` SYCL context when submitting for execution programs that would access this
allocation.

Since ``sycl::context`` is dynamically constructed by each extension  sharing a USM allocation,
in general, requires sharing the ``sycl::context`` along with the USM pointer, as it is done
in ``__sycl_usm_array_interface__`` :ref:`attribute <suai_attribute>`.

Since DLPack itself does not provide for storing of the ``sycl::context``, the proper
working of :func:`dpctl.tensor.from_dlpack` function is only supported for devices of those
platforms that support default platform context SYCL extension `sycl_ext_oneapi_default_platform_context`_,
and only of those allocations that are bound to this default context.

To query where a particular device ``dev`` belongs to a platform that implements
the default context, check whether ``dev.sycl_platform.default_context`` returns an instance
of :class:`dpctl.SyclContext` or raises an exception.


.. _sycl_ext_oneapi_default_platform_context: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_default_context.asciidoc


.. _user_guide_queues:

Queue
-----

SYCL queue is an entity associated with scheduling computational tasks for execution
on a targeted SYCL device and using some specific SYCL context.

Queue constructor generally requires both to be specified. For platforms that support the
default platform context, a shortcut queue constructor call that specifies only a device would
use the default platform context associated with the platform given device is a part of.

.. code-block:: python
   :caption: Queues constructed from device instance or filter string that selects it have the same context

   >>> import dpctl
   >>> d = dpctl.SyclDevice("gpu")
   >>> q1 = dpctl.SyclQueue(d)
   >>> q2 = dpctl.SyclQueue("gpu")
   >>> q1.sycl_context == q2.sycl_context, q1.sycl_device == q2.sycl_device
   (True, True)
   >>> q1 == q2
   False

Even through ``q1`` and ``q2`` instances of :class:`dpctl.SyclQueue` target the same device and use the same context
they do not compare equal, since they correspond to two independent scheduling entities.

.. note::
   :class:`dpctl.tensor.usm_ndarray` objects one associated with ``q1`` and another associated with ``q2``
   could not be combined in a call to the same function that implements
   :ref:`compute-follows-data <dpctl_tensor_compute_follows_data>` programming model in :mod:`dpctl.tensor`.


Event
-----

A SYCL event is an entity created when a task is submitted to SYCL queue for execution. The events are used to
order execution of computational tasks by the DPC++ runtime. They may also contain profiling information associated
with the submitted task, provided the queue was created with "enable_profiling" property.

SYCL event can be used to synchronize execution of the associated task with execution on host by using
:meth:`dpctl.SyclEvent.wait`.

Methods :meth:`dpctl.SyclQueue.submit_async` and :meth:`dpctl.SyclQueue.memcpy_async` return
:class:`dpctl.SyclEvent` instances.

.. note::
   At this point, :mod:`dpctl.tensor` does not provide public API for accessing SYCL events associated with
   submission of computation tasks implementing operations on :class:`dpctl.tensor.usm_ndarray` objects.


Unified Shared Memory
---------------------

Unified Shared Memory allocations of each kind are represented through Python classes
:class:`dpctl.memory.MemoryUSMDevice`, :class:`dpctl.memory.MemoryUSMShared`, and
:class:`dpctl.memory.MemoryUSMHost`.

These class constructors allow to make USM allocations of requested size in bytes
on the devices targeted by given SYCL queue, and are bound to the context from that
queue. This queue argument is stored the instance of the class and is used to submit
tasks to when performing copying of elements from or to this allocation or when filling
the allocation with values.

Classes that represent host-accessible USM allocations, i.e., types USM-shared and USM-host,
expose Python buffer interface.

.. code-block:: python

   >>> import dpctl.memory as dpm
   >>> import numpy as np

   >>> # allocate USM-shared memory for 6 32-bit integers
   >>> mem_d = dpm.MemoryUSMDevice(26)
   >>> mem_d.copy_from_host(b"abcdefghijklmnopqrstuvwxyz")

   >>> mem_s = dpm.MemoryUSMShared(30)
   >>> mem_s.memset(value=ord(b"-""))
   >>> mem_s.copy_from_device(mem_d)

   >>> # since USM-shared is host-accessible,
   >>> # it implements Python buffer protocol that allows
   >>> # for Python objects to read this USM allocation
   >>> bytes(mem_s)
   b'abcdefghijklmnopqrstuvwxyz--'


Backend
-------

Intel(R) oneAPI Data Parallel C++ compiler ships with two backends:

#. OpenCL* backend
#. Level-Zero backend

Additional backends can be added to the compiler by installing CodePlay's plugins:

#. CUDA backend: provided by `oneAPI for NVIDIA(R) GPUs <codeplay_nv_plugin_>`_ from `CodePlay`_
#. HIP backend: provided by `oneAPI for AMD GPUs <codeplay_amd_plugin_>`_ from `CodePlay`_

.. _codeplay_nv_plugin: https://developer.codeplay.com/products/oneapi/nvidia/
.. _codeplay_amd_plugin: https://developer.codeplay.com/products/oneapi/amd/
.. _CodePlay: https://codeplay.com/

When building open source `Intel LLVM <InteLlVmGh_>`_ compiler from source the project can be
configured to enable different backends (see `Get Started Guide <GetStartedGuide_>`_ for
further details).

.. _GetStartedGuide: https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md
.. _InteLlVmGh: https://github.com/intel/llvm
