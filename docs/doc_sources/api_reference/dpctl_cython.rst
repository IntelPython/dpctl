.. _dpctl_cython_api:

:py:mod:`dpctl` Cython API
==========================

.. role:: python(code)
   :language: python

All Python modules of :py:mod:`dpctl` come with ``__init__.pxd`` alongside ``__init__.py`` files
permitting doing both :python:`import dpctl` and :code:`cimport dpctl as c_dpctl`.

Locations of Cython declaration files in the package installation layout are as follows:

.. code-block:: text

    __init__.pxd
    _backend.pxd
    _sycl_context.pxd
    _sycl_device.pxd
    _sycl_device_factory.pxd
    _sycl_event.pxd
    _sycl_platform.pxd
    _sycl_queue.pxd
    _sycl_queue_manager.pxd
    sycl.pxd

    memory/__init__.pxd
    memory/_memory.pxd

    program/__init__.pxd
    program/_program.pxd

    tensor/__init__.pxd
    tensor/_usmarray.pxd
    tensor/_dlpack.pxd

File ``_backend.pxd`` redefines symbols from :ref:`DPCTLSyclInterface library <libsyclinterface>` for Cython.

File ``sycl.pxd`` provides casters from opaque types in "DPCTLSyclInterface" C library to SYCL C++ object pointers.

Please refer to the `examples/cython <https://github.com/IntelPython/dpctl/blob/master/examples/cython>`_ folder in the project
repository for a collection of examples.
