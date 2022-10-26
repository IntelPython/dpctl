Welcome to the Data-parallel Control (dpctl) Documentation!
===========================================================

The data-parallel control (dpctl) library provides C and Python bindings for
:sycl_spec_2020:`SYCL 2020 <>`. The SYCL 2020 features supported by dpctl are
limited to those included by Intel(R) DPCPP compiler and specifically cover the
SYCL runtime classes described in :sycl_runtime_classes:`Section 4.6 <>`
of the SYCL 2020 specification.

Apart from the bindings for these runtime
classes, dpctl includes bindings for SYCL USM memory allocators and
deallocators. Dpctl Python API provides classes that implement
`Python buffer protocol <https://docs.python.org/3/c-api/buffer.html>`_
using SYCL USM memory. It makes it possible to create Python objects that are
backed by SYCL USM memory.

Dpctl also supports the DPCPP :oneapi_filter_selection:`oneapi::filter_selector <>` extension and has
experimental support for SYCL's ``kernel`` and ``program`` classes.
