.. _filter_selector_string:

Filter Selector String
======================

Filter selector string is a way to refer to unpartitioned SYCL devices.
:mod:`dpctl` implements the syntax of the DPC++
:oneapi_filter_selection:`sycl_ext_oneapi_filter_selector <>` extension
without relying on the extension itself.

A filter selector string defines one or more filters, which must be
separated using ``","`` character. A filter is specified as a
triple of the form:

.. code-block:: text

    Backend:DeviceType:RelativeDeviceNumber

Every element of the triple is optional and the elements may be given in
any order.

``Backend`` specifies the desired backend of targeted devices, while
``DeviceType`` specifies the type of targeted devices.
``RelativeDeviceNumber`` refers to the number of the device that matches
any other given requirements, starting from 0 to marking the
"first device that matches the requirements".

If several devices match, the one with the highest default selector
score is selected.

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
:py:class:`dpctl.SyclPlatform`, or :py:class:`dpctl.SyclQueue`
classes.
