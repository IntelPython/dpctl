.. _libsyclinterface:

C Library SyclInterface
=======================

:py:mod:`dpctl` contains SyclInterface library that provides C API to a subset of functionality exposed in DPC++ runtime classes.

The C-API was necessary to support :py:mod:`numba_dpex` project to use DPC++ runtime classes from LLVM it generates.

Full :doc:`API reference <generated/index>` is generated using doxyrest from doxygen strings.

.. toctree::
    :hidden:

    generated/index
