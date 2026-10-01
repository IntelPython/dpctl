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
    :meth:`dpctl.SyclQueue.keep_args_alive`.

    Record events that gate the release of objects used by a task with
    ``add_cleanup_event``, and find them in ``cleanup_events``. They are
    waited on, but never become dependencies of later tasks.

    Waiting with ``wait`` also drops the references that
    :meth:`dpctl.SyclQueue.keep_args_alive` took for tasks that have since
    completed, which ``dpctl`` otherwise does on a thread of its own.

    .. deprecated:: 0.23.0
        ``add_event_pair``, ``host_task_events`` and ``num_host_task_events``
        are deprecated. Tasks are no longer paired with a host task event, so
        ``add_event`` takes the computational event alone, and cleanup is
        tracked by ``add_cleanup_event``, ``cleanup_events`` and
        ``num_cleanup_events``.
