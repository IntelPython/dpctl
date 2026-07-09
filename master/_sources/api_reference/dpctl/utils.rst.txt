.. _dpctl_utils_pyapi:

:py:mod:`dpctl.utils`
=====================

.. py:module:: dpctl.utils

.. currentmodule:: dpctl.utils

.. autofunction:: onetrace_enabled

.. autofunction:: intel_device_info

.. data:: SequentialOrderManager

    Thread-local object mapping each :class:`dpctl.SyclQueue` to an order
    manager, used to ensure sequential ordering of offloaded tasks.
