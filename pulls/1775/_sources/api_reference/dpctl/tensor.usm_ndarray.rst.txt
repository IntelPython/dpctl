.. _dpctl_tensor_array_object:

USM array object
================

.. currentmodule:: dpctl.tensor

The array object represents a multi-dimensional tensor of uniform elemental datatype allocated on
a :py:class:`Device`. The tensor in stored in a USM allocation, which can be accessed via
:py:attr:`usm_ndarray.base` attribute.

Implementation of :py:class:`usm_ndarray` conforms to
`Array API standard <array_api_array_object>`_ specification.

.. array_api_array_object: https://data-apis.org/array-api/latest/API_specification/array_object.html

.. autosummary::
    :toctree: generated
    :template: autosummary/usm_ndarray.rst

    usm_ndarray

.. _dpctl_tensor_usm_ndarray_to_device_example:

.. include:: examples/usm_ndarray.rst
