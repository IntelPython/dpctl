.. _api_reference:

=============
API Reference
=============

The package ``dpctl`` provides

* Python language bindings for the DPC++ runtime
   - :ref:`API objects <dpctl_pyapi>` in :py:mod:`dpctl` namespace
   - :ref:`API objects <dpctl_memory_pyapi>` in :py:mod:`dpctl.memory` namespace
   - :ref:`API objects <dpctl_program_pyapi>` in :py:mod:`dpctl.program` namespace
   - :ref:`API objects <dpctl_utils_pyapi>` in :py:mod:`dpctl.utils` namespace
* SYCL-based Python array library
   - :ref:`API objects <dpctl_tensor_pyapi>` in :py:mod:`dpctl.tensor` namespace
* Python C-API
   - :ref:`C API <dpctl_capi>` for working with Python classes defined in :mod:`dpctl`
* Extension generators
   - :ref:`Cython declarations <dpctl_cython_api>` for classes defined in :py:mod:`dpctl` and supporting functions
   - :ref:`Integration with pybind11 <dpctl_pybind11_api>`, defining type casters mapping SYCL classes to Python classes, as well as defining C++ classes wrapping a Python object for :class:`dpctl.tensor.usm_ndarray` and :mod:`dpctl.memory` objects.
   - :ref:`Integration with CMake <dpctl_cmake_support>` to simplify building DPC++-based Python extension using scikit-build.
* C API for DPC++ runtime
   - :doc:`DPCTLSyclInterface C library <libsyclinterface/generated/index>`

.. toctree::
   :hidden:

   dpctl/index
   dpctl/memory
   dpctl/program
   dpctl/utils
   dpctl/tensor
   libsyclinterface/index
   dpctl_capi
   dpctl_cython
   dpctl_pybind11
   dpctl_cmake
