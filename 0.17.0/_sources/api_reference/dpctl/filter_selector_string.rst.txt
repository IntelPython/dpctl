.. _filter_selector_string:

Filter Selector String
======================

Filter selector string is a way to refer to unpartitioned SYCL devices
proposed in :oneapi_filter_selection:`sycl_ext_oneapi_filter_selector <>`
extension to SYCL standard.

This document captures aspects of the referenced document relevant
to :mod:`dpctl`.

A filter selector string defines one or more filters, which must be
separated using ``","`` character. A filter is specified as a
triple of the form:

.. code-block:: text

    Backend:DeviceType:RelativeDeviceNumber

Every element of the triple is optional, but a filter must contain at
least one component.

``Backend`` specifies the desired backend of targeted devices, while
``DeviceType`` specifies the type of targeted devices.
``RelativeDeviceNumber`` refers to the number of the device that matches
any other given requirements, starting from `0` to marking the
"first device that matches the requirements".

Attempting to use a non-conforming string in places where filter selector
string is expected will raise an exception.

Supported values for ``Backend`` are:

.. list-table::

    * - cuda
      - opencl
      - level_zero
      - hip

Supported values for ``DeviceType`` are:

.. list-table::

    * - accelerator
      - cpu
      - gpu

Filter selector strings can be used as arguments to constructors of
:py:class:`dpctl.SyclDevice`, :py:class:`dpctl.SyclContext`,
:py:class:`dpctl.SyclPlatform`, :py:class:`dpctl.SyclQueue`,
or :py:class:`dpctl.tensor.Device` classes, as well as values of
``device`` keyword in :ref:`array creation functions <dpctl_tensor_creation_functions>`.
