.. _querying_platforms:

########
Platform
########

A platform abstracts a device driver for one or more XPU that is connected to
a host. The :class:`dpctl.SyclPlatform` class represents a platform and
abstracts the :sycl_platform:`sycl::platform <>` SYCL runtime class.

Listing Available Platforms
---------------------------

The platforms available on a system can be queried using the
:func:`dpctl.lsplatform` function. In addition, as illustrated in the following
example it is possible to print out metadata about a platform.

.. literalinclude:: ../../../../../examples/python/lsplatform.py
    :language: python
    :lines: 20-41
    :linenos:

The example can be executed as follows:

.. code-block:: bash

    python dpctl/examples/python/lsplatform.py -r all

The possible output for the example may be:

.. program-output:: python ../examples/python/lsplatform.py -r all

.. Note::
    The verbosity for the output can be controlled using the ``verbosity``
    keyword argument. Refer :func:`dpctl.lsplatform`.
