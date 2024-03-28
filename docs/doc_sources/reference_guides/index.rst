.. _reference_guides:

================
Reference guides
================

The package ``dpctl`` provides

* Python API
   - :ref:`API objects <dpctl_pyapi>` in :py:mod:`dpctl` namespace
   - :ref:`API objects <dpctl_tensor_pyapi>` in :py:mod:`dpctl.tensor` namespace
   - :ref:`API objects <dpctl_memory_pyapi>` in :py:mod:`dpctl.memory` namespace
   - :ref:`API objects <dpctl_program_pyapi>` in :py:mod:`dpctl.program` namespace
   - :ref:`API objects <dpctl_utils_pyapi>` in :py:mod:`dpctl.utils` namespace
* :ref:`C API <dpctl_capi>` for working with Python classes defined in :mod:`dpctl`
* Cython declarations for these classes
* Integration with pybind11, defining type casters mapping SYCL classes to Python classes, as well as defining C++ classes wrapping a Python object for :class:`dpctl.tensor.usm_ndarray` and :mod:`dpctl.memory` objects.
* SyclInterface C library for working in DPC++ runtime objects from C
* Integration with CMake to simplify building DPC++-based Python extension using scikit-build.

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
