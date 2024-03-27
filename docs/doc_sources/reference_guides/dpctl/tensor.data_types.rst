.. _dpctl_tensor_data_types:

.. currentmodule:: dpctl.tensor

Data types
==========

:py:mod:`dpctl.tensor` supports the following data types:

+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Data Type      | Description                                                                                                                                                                             |
+================+=========================================================================================================================================================================================+
| ``bool``       | Boolean (``True`` or ``False``)                                                                                                                                                         |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``int8``       | An 8-bit signed integer type capable of representing :math:`v` subject to :math:`-2^7 \le v < 2^7`                                                                                      |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``int16``      | A 16-bit signed integer type capable of representing :math:`v` subject to :math:`-2^{15} \le v < 2^{15}`                                                                                |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``int32``      | A 32-bit signed integer type capable of representing :math:`v` subject to :math:`-2^{31} \le v < 2^{31}`                                                                                |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``int64``      | A 64-bit signed integer type capable of representing :math:`v` subject to :math:`-2^{63} \le v < 2^{63}`                                                                                |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``uint8``      | An 8-bit unsigned integer type capable of representing :math:`v` subject to :math:`0 \le v < 2^8`                                                                                       |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``uint16``     | A 16-bit unsigned integer type capable of representing :math:`v` subject to :math:`0 \le v < 2^{16}`                                                                                    |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``uint32``     | A 32-bit unsigned integer type capable of representing :math:`v` subject to :math:`0 \le v < 2^{32}`                                                                                    |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``uint64``     | A 64-bit unsigned integer type capable of representing :math:`v` subject to :math:`0 \le v < 2^{64}`                                                                                    |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``float16``    | An IEEE-754 half-precision (16-bits) binary floating-point number (see `IEEE 754-2019`_)                                                                                                |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``float32``    | An IEEE-754 single-precision (32-bits) binary floating-point number (see `IEEE 754-2019`_)                                                                                              |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``float64``    | An IEEE-754 double-precision (64-bits) binary floating-point number (see `IEEE 754-2019`_)                                                                                              |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``complex64``  | Single-precision (64-bit) complex floating-point number whose real and imaginary components are IEEE 754 single-precision (32-bit) binary floating-point numbers (see `IEEE 754-2019`_) |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``complex128`` | Double-precision (128-bit) complex floating-point number whose real and imaginary components are IEEE 754 double-precision (64-bit) binary floating-point numbers (see `IEEE 754-2019`_)|
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

.. _IEEE 754-2019: https://doi.org/10.1109%2FIEEESTD.2019.8766229

Data type support by array object :py:class:`usm_ndarray` depends on capabilities of :class:`dpctl.SyclDevice` where array is allocated.

Half-precision floating-point type ``float16`` is supported only for devices whose attribute :attr:`dpctl.SyclDevice.has_aspect_fp16` evaluates to ``True``.

Double-precision floating-point type ``float64`` and double-precision complex floating-point type ``complex128`` are supported only for devices whose attribute :attr:`dpctl.SyclDevice.has_aspect_fp64`
evaluates to ``True``.

Requests to create an instance of array object for these types on devices where these prerequisites are not met will raise an exception.

.. TODO: provide a note on support for sub-normal numbers

Data type objects are instances of :py:class:`numpy.dtype` object, and support equality comparison by implementing
special method :meth:`__eq__`.
