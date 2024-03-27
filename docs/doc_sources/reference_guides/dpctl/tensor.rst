.. _dpctl_tensor_pyapi:

:py:mod:`dpctl.tensor`
======================

.. py:module:: dpctl.tensor

.. currentmodule:: dpctl.tensor

:py:mod:`dpctl.tensor` provides a reference implementation of
:array_api:`Python Array API <>` specification. The implementation
uses :ref:`data-parallel <parallelism_definitions>` algorithms suitable for execution on accelerators,
such as GPUs.

:py:mod:`dpctl.tensor` is written using C++ and :sycl_spec_2020:`SYCL <>`
and oneAPI extensions implemented in :dpcpp_compiler:`Intel(R) oneAPI DPC++ compiler <>`.

This module contains:

* Array object :py:class:`usm_ndarray`
* :ref:`array creation functions <dpctl_tensor_creation_functions>`
* :ref:`array manipulation functions <dpctl_tensor_manipulation_functions>`
* :ref:`elementwise functions <dpctl_api_elementwise_functions>`
* :ref:`indexing functions <dpctl_tensor_indexing_functions>`
* :ref:`introspection functions <dpctl_tensor_inspection>`
* :ref:`searching functions <dpctl_tensor_searching_functions>`
* :ref:`set functions <dpctl_tensor_set_functions>`
* :ref:`sorting functions <dpctl_tensor_sorting_functions>`
* :ref:`statistical functions <dpctl_tensor_statistical_functions>`
* :ref:`utility functions <dpctl_tensor_utility_functions>`


.. toctree::
    :hidden:

    tensor.creation_functions
    tensor.usm_ndarray
    tensor.data_type_functions
    tensor.data_types
    tensor.elementwise_functions
    tensor.indexing_functions
    tensor.inspection
    tensor.manipulation_functions
    tensor.searching_functions
    tensor.set_functions
    tensor.sorting_functions
    tensor.statistical_functions
    tensor.utility_functions
