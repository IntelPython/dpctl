.. _platforms:

########
Platform
########

A platform abstracts a device driver for one or more XPUs that is connected to
a host. The :class:`dpctl.SyclPlatform` class represents a platform and
abstracts the :sycl_platform:`sycl::platform <>` SYCL runtime class.

Listing Available Platforms
---------------------------

To require the platforms available on a system, use
:func:`dpctl.lsplatform` function.

It is possible to print out metadata about a platform:

.. literalinclude:: ../../../../../examples/python/lsplatform.py
    :language: python
    :lines: 20-41
    :linenos:

To execute the example, run:

.. code-block:: bash

    python dpctl/examples/python/lsplatform.py -r all

The possible output for the example:

.. program-output:: python ../examples/python/lsplatform.py -r all

.. Note::
    To control the verbosity for the output, use the ``verbosity``
    keyword argument. Refer to :func:`dpctl.lsplatform` for more information.
