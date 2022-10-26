Welcome to the Data-parallel Control (dpctl) Documentation!
===========================================================

The data-parallel control (dpctl) library provides C and Python bindings for
:sycl_spec_2020:`SYCL 2020 <>`. The SYCL 2020 features supported by dpctl are
limited to those included by Intel(R) DPC++ compiler and specifically cover the
SYCL runtime classes described in :sycl_runtime_classes:`Section 4.6 <>`
of the SYCL 2020 specification.

Apart from the bindings for these runtime
classes, dpctl includes bindings for SYCL USM memory allocators and
deallocators. Dpctl Python API provides classes that implement
`Python buffer protocol <https://docs.python.org/3/c-api/buffer.html>`_

Dpctl implements N-dimensional array Python object :class:`dpctl.tensor.usm_ndarray` to
logically address USM memory allocations, and contains growing implementation of
operations on such objects confofrming :array_api:`array API specification <>` in
:class:`dpctl.tensor` namespace.

Dpctl also supports the DPCPP :oneapi_filter_selection:`oneapi::filter_selector <>` extension and has
experimental support for SYCL's interoperability ``kernel`` and
``kernel_bundle<bundle_state::executable>`` classes.
