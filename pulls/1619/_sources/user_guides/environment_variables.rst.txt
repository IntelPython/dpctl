.. _user_guides_env_variables:

#####################
Environment variables
#####################

Behavior of :py:mod:`dpctl` is affected by :dpcpp_envar:`environment variables <>` that
affect DPC++ compiler runtime.

Variable ``ONEAPI_DEVICE_SELECTOR``
-----------------------------------

The varible ``ONEAPI_DEVICE_SELECTOR`` can be used to limit the choice of devices
available to :py:mod:`dpctl`. Please refer to
:ref:`Managing Devices <beginners_guide_oneapi_device_selector>` for detailed
description and uses.

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
