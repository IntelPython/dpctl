.. _beginners_guide_env_variables:

Environment variables
=====================

Behavior of :py:mod:`dpctl` is affected by :dpcpp_envar:`environment variables <>` that
affect DPC++ compiler runtime. Particularly, the varible ``ONEAPI_DEVICE_SELECTOR`` can be
used to limit the choice of devices available to :py:mod:`dpctl`.

As such, the device returned by :py:func:`select_default_device`, as well the behavior
of default constructor of :py:class:`SyclDevice` class is infuenced by settings of this
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
