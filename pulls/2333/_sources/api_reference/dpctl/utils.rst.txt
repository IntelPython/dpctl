.. _dpctl_utils_pyapi:

:py:mod:`dpctl.utils`
=====================

.. py:module:: dpctl.utils

.. currentmodule:: dpctl.utils

.. autofunction:: onetrace_enabled

.. autofunction:: intel_device_info

.. data:: SequentialOrderManager

    Thread-local instance of
    :class:`~dpctl.utils._order_manager.SyclQueueToOrderManagerMap` used to
    ensure sequential ordering of tasks offloaded to a :class:`dpctl.SyclQueue`.

.. autoclass:: dpctl.utils._order_manager.SyclQueueToOrderManagerMap
    :members:
