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

    Record submitted tasks with ``add_event`` and use ``submitted_events``
    as the dependency list of subsequent submissions. To keep Python objects
    referenced by a task alive until it completes, use
    :func:`dpctl.keep_args_alive`.

    .. deprecated:: 0.23.0
        ``add_event_pair``, ``host_task_events`` and ``num_host_task_events``
        are deprecated. Tasks are no longer paired with a host task event,
        so ``add_event`` takes the computational event alone.
