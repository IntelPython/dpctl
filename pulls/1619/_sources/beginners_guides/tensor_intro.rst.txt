.. _beginners_guide_tensor_intro:

Getting started with :py:mod:`dpctl.tensor`
===========================================

The tensor submodule provides an N-dimensional array object for a tensor whose values have the same data type
from the :ref:`following list <dpctl_tensor_data_types>`:

.. currentmodule:: dpctl.tensor

.. list-table::

    * -
      - :attr:`int8`
      - :attr:`int16`
      - :attr:`int32`
      - :attr:`int64`
      -
      - :attr:`float16`
      - :attr:`float32`
      - :attr:`complex64`

    * - :attr:`bool`
      - :attr:`uint8`
      - :attr:`uint16`
      - :attr:`uint32`
      - :attr:`uint64`
      -
      -
      - :attr:`float64`
      - :attr:`complex128`


Creating an array
-----------------

Array :ref:`creation functions <dpctl_tensor_creation_functions>` support keyword arguments that
control the device where the array is allocated as well as aspects of
:ref:`Unified Shared Memory allocation <dpctl_memory_pyapi>` for the array.

These three keywords are:

.. list-table::
    :header-rows: 1

    * - Keyword arguments
      - Default value
      - Description
    * - ``usm_type``
      - ``"device"``
      - type of USM allocation to make
    * - ``device``
      - ``None``
      - :py:class:`dpctl.tensor.Device` instance
    * - ``sycl_queue``
      - ``None``
      - Instance of :class:`dpctl.SyclQueue` associated with array

Arguments ``sycl_queue`` and ``device`` are complementary to each other, and
a user need only provide one of these.

A valid setting for the ``device`` keyword argument is any object that can be passed to :py:meth:`dpctl.tensor.Device.create_device`.
If both ``device`` and ``sycl_queue`` keyword arguments are specified, they must correspond to :class:`dpctl.SyclQueue` instances which
compare equal to one another.

A created instance of :class:`usm_ndarray` has an associated :class:`dpctl.SyclQueue` instance that can be retrieved
using :attr:`dpctl.tensor.usm_ndarray.sycl_queue` property. The underlying USM allocation
is allocated on :class:`dpctl.SyclDevice` and is bound to :class:`dpctl.SyclContext` targeted by this queue.


Execution model
---------------

.. _dpctl_tensor_compute_follows_data:

When one of more instances of ``usm_ndarray`` objects are passed to a function in :py:mod:`dpctl.tensor` other than creation function,
a "compute follows data" execution model is followed.

The model requires that :class:`dpctl.SyclQueue` instances associated with each array compared equal to one another, signifying that
each one corresponds to the same underlying ``sycl::queue`` object. In such a case, the output array is associated with the same
``sycl::queue`` and computations are scheduled for execution using this ``sycl::queue``.

.. note::
    Two instances :class:`dpctl.SyclQueue` may target the same ``sycl::device`` and be using the same ``sycl::context``, but correspond
    to different scheduling enties, and hence be in violation of the compute-follows-data requirement. One common example of this are
    ``SyclQueue`` corresponding to default-selected device and using platform default context but created using different properties, e.g.
    one with `"enable_profiling"` set and another without it.

If input arrays do not conform to the compute-follows-data requirements, :py:exc:`dpctl.utils.ExecutionPlacementError` is raised.
User must explicitly migrate the data to unambiguously control the execution placement.


Migrating arrays
----------------

Array content can be migrated to a different device :ref:`using <dpctl_tensor_usm_ndarray_to_device_example>`
either :meth:`dpctl.tensor.usm_ndarray.to_device` method, or by using :func:`dpctl.tensor.asarray` function.

The ``arr.to_device(device=target_device)`` method will be zero-copy if the ``arr.sycl_queue`` and the :class:`dpctl.SyclQueue`
instance associated with new target device have the same underlying ``sycl::device`` and ``sycl::context`` instances.

Here is an example of migration without a copy:

.. code-block:: python
    :caption: Using ``to_device`` to zero-copy migrate array content to be associated with a different ``sycl::queue``

    import dpctl
    from dpctl import tensor

    x = tensor.linspace(0, 1, num=10**8)
    q_prof = dpctl.SyclQueue(x.sycl_context, x.sycl_device, property="enable_profiling")

    timer = dpctl.SyclTimer()
    # no data migration takes place here,
    # but x and x1 arrays do not satify compute-follows-data requirements
    x1 = x.to_device(q_prof)

    with timer(q_prof):
        y = tensor.sin(2*x1)*tensor.exp(-tensor.square(x1))

    host_dt, device_dt = timer.dt
    print(f"Execution on device {x.sycl_device.name} took {device_dt} seconds, on host {host_dt} seconds")
