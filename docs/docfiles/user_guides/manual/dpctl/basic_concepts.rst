.. _basic_concepts:

Basic Concepts
==============

This section introduces the basic concepts for XPU management used by `dpctl`.

.. note:: For SYCL-level details, refer to a more topical SYCL* reference, such as the :sycl_spec_2020:`SYCL 2020 spec <>`.

* **Heterogeneous computing**
    Refers to using multiple devices in a program.

* **Host**
    Every program starts by running on a host, and most of the lines of code in
    a program, in particular lines of code implementing the Python interpreter
    itself, are usually for the host. Hosts are customarily CPUs.

* **Device**
    A device is an XPU connected to a host that is programmable with a specific
    device driver. Different types of devices can have different architectures
    (CPUs, GPUs, FPGA, ASICs, DSP) but are programmable using the same
    :oneapi:`oneAPI <>` programming model.

* **Platform**
    A device driver installed on the system is called the platform. As multiple
    devices of the same type can share the same device driver, a platform may
    contain multiple devices. The same physical hardware (for example, GPU)
    may be reflected as two separate devices if they can be programmed by more
    than one platform. For example, the same GPU hardware can be listed as an
    OpenCL* GPU device and a Level-Zero* GPU device.

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
   Holds information related to computation or data movement operation
   scheduled for the execution on a queue. An event can store the execution status or
   profiling information of the queue, to which the task is submitted.
   Events can be used to specify task
   dependencies or to synchronize host and devices.

* **USM**
   Unified Shared Memory (USM) refers to pointer-based device memory management.
   USM allocations are bound to context. It means, a pointer representing
   USM allocation can be unambiguously mapped to the data it represents only
   if the associated context is known. USM allocations are accessible by
   computational kernels that are executed on a device, provided that the
   allocation is bound to the same context that is used to construct the queue
   where the kernel is scheduled for execution.

   Depending on the capability of the device, USM allocations can be:

.. csv-table::
   :header: "Name", "Allocation accessible", "Access"
   :widths: 25, 25, 50

   "Device allocation", "No","Refers to an allocation in host memory that is accessible from a device."
   "Shared allocation", "Yes", "Accessible by both the host and device."
   "Host allocation", "Yes", "Accessible by both the host and device."


Runtime manages synchronization of the host's and device's view into shared allocations.
The initial placement of the shared allocations is not defined.

* **Backend**
   Refers to the implementation of :oneapi:`oneAPI <>` programming model exposed
   by the underlying runtime.
