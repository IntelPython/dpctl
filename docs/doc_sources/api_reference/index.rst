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
   - :ref:`Declarations <dpctl_cython_api>` for classes defined in :py:mod:`dpctl` and supporting functions for use in `Cython <cython_docs_>`_.
   - :ref:`Integration <dpctl_pybind11_api>` with `pybind11 <pybind11_docs_>`_ defines type casters connecting SYCL classes and Python classes, as well as defines C++ classes wrapping a Python object for :class:`dpctl.tensor.usm_ndarray` and :mod:`dpctl.memory` objects.
   - :ref:`Integration <dpctl_cmake_support>` with `CMake <cmake_docs_>`_ to simplify building DPC++-based Python extension using `scikit-build <skbuild_docs_>`_.
* C API for DPC++ runtime
   - :doc:`DPCTLSyclInterface C library <libsyclinterface/generated/index>`


.. _cmake_docs: https://cmake.org/documentation/
.. _cython_docs: https://cython.readthedocs.io/en/latest/
.. _skbuild_docs: https://scikit-build.readthedocs.io/en/latest/
.. _pybind11_docs: https://pybind11.readthedocs.io/en/stable/

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
