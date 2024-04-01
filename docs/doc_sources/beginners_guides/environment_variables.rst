.. _beginners_guide_env_variables:

Environment variables
=====================

Behavior of :py:mod:`dpctl` is affected by :dpcpp_envar:`environment variables <>` that
affect DPC++ compiler runtime.

Variable ``ONEAPI_DEVICE_SELECTOR``
-----------------------------------

The varible ``ONEAPI_DEVICE_SELECTOR`` can be
used to limit the choice of devices available to :py:mod:`dpctl`.

As such, the device returned by :py:func:`select_default_device`, as well the behavior
of default constructor of :py:class:`SyclDevice` class is influenced by settings of this
variable.

Some users may find it convenient to also use a default-selected device, but control
which device that may be using ``ONEAPI_DEVICE_SELECTOR``. For example, the following script:

.. code-block::python
    :caption: Sample array computation script "run.py"

    from dpctl import tensor as dpt

    gamma = 0.34
    x = dpt.linspace(0, 2*dpt.pi, num=10**6)
    f = dpt.sin(gamma * x) * dpt.exp(-x)

    int_approx = dpt.sum(f)
    print(f"Approximate value of integral: {int_approx} running on {x.device}" )

This script may be executed on a CPU, or GPU as follows:

.. code-block:: bash

    # execute on CPU device
    ONEAPI_DEVICE_SELECTOR=*:cpu python run.py
    #   Output: Approximate value of integral: 48328.99708167 running on Device(opencl:cpu:0)

    # execute on GPU device
    ONEAPI_DEVICE_SELECTOR=*:gpu python run.py
    #   Output: Approximate value of integral: 48329. running on Device(level_zero:gpu:0)


Variable ``SYCL_CACHE_PERSISTENT``
----------------------------------

The binaries implementing :py:mod:`dpctl.tensor` created using DPC++ compiler contain sections
with standardized intermediate forms (e.g. `SPIR-V <https://www.khronos.org/spir/>`_) that must be
further built using SYCL device drivers for execution on the specific target hardware.
This step is known as just-in-time compiling (JIT-ing).

By default, the result of JIT-ing persists for the duration of SYCL application, i.e. for the
duration of the Python session where :py:mod:`dpctl.tensor` is used. Setting environment variable
``SYCL_CACHE_PERSISTENT`` to value of ``1`` instructs DPC++ runtime to save the result of JIT-ing to
disk and reuse it in subsequent Python sessions (assuming the variable remains to be set when sessions
are started).

Setting of the environment variable ``SYCL_CACHE_PERSISTENT`` improves times of function invocations,
but requires sufficient disk space.
