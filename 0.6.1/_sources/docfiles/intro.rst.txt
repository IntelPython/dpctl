Welcome to Data-parallel Control (dpctl)'s documentation!
=========================================================

The data-parallel control (dpctl) library provides C and Python bindings for
`SYCL 2020 <https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html>`_.
The SYCL 2020 features supported by dpctl are limited to those included by
Intel's DPCPP compiler and specifically cover the SYCL runtime classes described
in `Section 4.6 <https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#_sycl_runtime_classes>`_
of the SYCL 2020 specification. Apart from the bindings for these runtime
classes, dpctl includes bindings for SYCL USM memory allocators and
deallocators. Dpctl's Python API provides classes that implement
`Python buffer protocol <https://docs.python.org/3/c-api/buffer.html>`_
using SYCL USM memory; making it possible to create Python objects that are
backed by SYCL USM memory.

Dpctl also supports the DPCPP ``ONEAPI::filter_selector`` extension and has
experimental support for SYCL's ``kernel`` and ``program`` classes.
