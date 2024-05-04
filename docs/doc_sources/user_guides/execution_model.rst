.. _oneapi_programming_model_intro:

########################
oneAPI programming model
########################

oneAPI library and its Python interface
=======================================

Using oneAPI libraries, a user calls functions that take ``sycl::queue`` and a collection of
``sycl::event`` objects among other arguments. For example:

.. code-block:: cpp
    :caption: Prototypical call signature of oneMKL function

    sycl::event
    compute(
        sycl::queue &exec_q,
        ...,
        const std::vector<sycl::event> &dependent_events
    );

The function ``compute`` inserts computational tasks into the queue ``exec_q`` for DPC++ runtime to
execute on the device the queue targets. The execution may begin only after other tasks whose
execution status is represented by ``sycl::event`` objects in the provided ``dependent_events``
vector complete. If the vector is empty, the runtime begins the execution as soon as the device is
ready. The function returns a ``sycl::event`` object representing completion of the set of
computational tasks submitted by the ``compute`` function.

Hence, in the oneAPI programming model, the execution **queue** is used to specify which device the
function will execute on. To create a queue, one must specify a device to target.

In :mod:`dpctl`, the ``sycl::queue`` is represented by :class:`dpctl.SyclQueue` Python type,
and a Python API to call such a function might look like

.. code-block:: python

    def call_compute(
        exec_q : dpctl.SyclQueue,
        ...,
        dependent_events : List[dpctl.SyclEvent] = []
    ) -> dpctl.SyclEvent:
        ...

Even if the Python API looks different from this to an offloading Python function, it must
translate to a similar call under the hood.

The arguments to the function must be suitable for use in the offloading functions.
Typically these are Python scalars, or objects representing USM allocations, such as
:class:`dpctl.tensor.usm_ndarray`, :class:`dpctl.memory.MemoryUSMDevice` and friends.

.. note::
    The USM allocations these objects represent must not get deallocated before
    offloaded tasks that access them complete.

    This is something authors of DPC++-based Python extensions must take care of,
    and users of such extensions should assume assured.


USM allocations in :mod:`dpctl` and compute-follows-data
=========================================================

To make a USM allocation on a device in SYCL, one needs to specify ``sycl::device`` in the
memory of which the allocation is made, and the ``sycl::context`` to which the allocation
is bound.

A ``sycl::queue`` object is often used instead. In such cases ``sycl::context`` and ``sycl::device`` associated
with the queue are used to make the allocation.

.. important::
    :mod:`dpctl` chose to associate a queue object with every USM allocation.

    The associated queue may be queried using ``.sycl_queue`` property of the
    Python type representing the USM allocation.

This design choice allows :mod:`dpctl` to have a preferred queue to use when operating on any single
USM allocation. For example:

.. code-block:: python

    def unary_func(x : dpctl.tensor.usm_ndarray):
        code1
        _ = _func_impl(x.sycl_queue, ...)
        code2

When combining several objects representing USM-allocations, the
:ref:`programming model <dpctl_tensor_compute_follows_data>`
adopted in :mod:`dpctl` insists that queues associated with each object be the same, in which
case it is the execution queue used. Alternatively :exc:`dpctl.utils.ExecutionPlacementError` is raised.

.. code-block:: python

    def binary_func(
        x1 : dpctl.tensor.usm_ndarray,
        x2 : dpctl.tensor.usm_ndarray
    ):
        exec_q = dpctl.utils.get_execution_queue((x1.sycl_queue, x2.sycl_queue))
        if exec_q is None:
            raise dpctl.utils.ExecutionPlacementError
        ...

In order to ensure that compute-follows-data works seamlessly out-of-the-box, :mod:`dpctl` maintains
a cache of with context and device as keys and queues as values used by :class:`dpctl.tensor.Device` class.

.. code-block:: python

    >>> import dpctl
    >>> from dpctl import tensor

    >>> sycl_dev = dpctl.SyclDevice("cpu")
    >>> d1 = tensor.Device.create_device(sycl_dev)
    >>> d2 = tensor.Device.create_device("cpu")
    >>> d3 = tensor.Device.create_device(dpctl.select_cpu_device())

    >>> d1.sycl_queue == d2.sycl_queue, d1.sycl_queue == d3.sycl_queue, d2.sycl_queue == d3.sycl_queue
    (True, True, True)

Since :class:`dpctl.tensor.Device` class is used by all :ref:`array creation functions <dpctl_tensor_creation_functions>`
in :mod:`dpctl.tensor`, the same value used as ``device`` keyword argument results in array instances that
can be combined together in accordance with compute-follows-data programming model.

.. code-block:: python

    >>> from dpctl import tensor
    >>> import dpctl

    >>> # queue for default-constructed device is used
    >>> x1 = tensor.arange(100, dtype="int32")
    >>> x2 = tensor.zeros(100, dtype="int32")
    >>> x12 = tensor.concat((x1, x2))
    >>> x12.sycl_queue == x1.sycl_queue, x12.sycl_queue == x2.sycl_queue
    (True, True)
    >>> # default constructors of SyclQueue class create different instance of the queue
    >>> q1 = dpctl.SyclQueue()
    >>> q2 = dpctl.SyclQueue()
    >>> q1 == q2
    False
    >>> y1 = tensor.arange(100, dtype="int32", sycl_queue=q1)
    >>> y2 = tensor.zeros(100, dtype="int32", sycl_queue=q2)
    >>> # this call raises ExecutionPlacementError since compute-follows-data
    >>> # rules are not met
    >>> tensor.concat((y1, y2))

Please refer to the :ref:`array migration <dpctl_tensor_array_migration>` section of the introduction to
:mod:`dpctl.tensor` to examples on how to resolve ``ExecutionPlacementError`` exceptions.

..
    Introduction
    ============

    :mod:`dpctl` leverages `Intel(R) oneAPI DPC++ compiler <dpcpp_compiler>`_ runtime to
    answer the following three questions users of heterogenous platforms ask:

    #.  What are available compute devices?
    #.  How to specify the device a computation is to be offloaded to?
    #.  How to manage sharing of data between devices and Python?

    :mod:`dpctl` implements Python classes and free functions mapping to DPC++
    entities to answer these questions.

    .. _dpcpp_compiler: https://www.intel.com/content/www/us/en/developer/tools/oneapi/data-parallel-c-plus-plus.html


    Available compute devices
    =========================

    Please refer to :ref:`managing devices <beginners_guide_managing_devices>` for details and examples of
    enumeration of available devices, as well as of selection of a particular device.

    Once a :class:`dpctl.SyclDevice` instance representing an underlying ``sycl::device`` is created,
    a :class:`dpctl.SyclQueue`

    The default behavior for creation functions in :mod:`dpctl.tensor` and constructors of USM allocation classes
    from :mod:`dpctl.memory` is to target the default-selected device (consistent with the behavior of SYCL-based
    C++ applications).

    .. code-block:: python

        >>> import dpctl
        >>> from dpctl import tensor
        >>> x = tensor.ones(777)
        >>> x.sycl_device == dpctl.select_default_device()
        True
        >>> from dpctl import memory
        >>> mem = memory.MemoryUSMDevice(80)
        >>> mem.sycl_device == dpctl.select_default_device()
        True

    For Python scripts that target only one device, it makes sense to always use the default-selected device, but
    :ref:`control <beginners_guide_oneapi_device_selector_usecase>` which device is being selected by DPC++ runtime
    as the default via ``ONEAPI_DEVICE_SELECTOR`` environment variable.

    Exacting device where computation occurs
    ========================================

    Sharing data between devices and Python
    =======================================

..
    The Data Parallel Control (:py:mod:`dpctl`) package provides a Python runtime to access a
    data-parallel computing resource (programmable processing units) from another Python application
    or a library, alleviating the need for the other Python packages to develop such a
    runtime themselves. The set of programmable processing units includes a diverse range of computing
    architectures such as a CPU, GPU, FPGA, and more. They are available to programmers on a
    modern heterogeneous system.

    The :py:mod:`dpctl` runtime is built on top of the C++ SYCL standard as implemented in
    `Intel(R) oneAPI DPC++ compiler <dpcpp_compiler>`_ and is designed to be both vendor and
    architecture agnostic.

    If the underlying SYCL runtime supports a type of architecture, the :mod:`dpctl` allows
    accessing that architecture from Python.

    In its current form, :py:mod:`dpctl` relies on certain DPC++ extensions of the
    SYCL standard. Moreover, the binary distribution of :py:mod:`dpctl` uses the proprietary
    Intel(R) oneAPI DPC++ runtime bundled as part of oneAPI and is compiled to only target
    Intel(R) XPU devices. :py:mod:`dpctl` supports compilation for other SYCL targets, such as
    ``nvptx64-nvidia-cuda`` and ``amdgcn-amd-amdhsa`` using `CodePlay plugins <codeplay_plugins_url_>`_
    for oneAPI DPC++ compiler providing support for these targets.

    :py:mod:`dpctl` is also compatible with the runtime of the `open-source DPC++ <os_intel_llvm_gh_url_>`_
    SYCL bundle that can be compiled to support a wide range of architectures including CUDA,
    AMD* ROC, and HIP*.

    The user guide introduces the core features of :py:mod:`dpctl` and the underlying
    concepts. The guide is meant primarily for users of the Python package. Library
    and native extension developers should refer to the programmer guide.

    .. _codeplay_plugins_url: https://developer.codeplay.com/products/oneapi/
    .. _os_intel_llvm_gh_url: https://github.com/intel/llvm
    .. _dpcpp_compiler: https://www.intel.com/content/www/us/en/developer/tools/oneapi/data-parallel-c-plus-plus.html
