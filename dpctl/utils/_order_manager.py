import sys
import threading
import weakref
from collections.abc import Sequence
from typing import Protocol

from .._sycl_event import SyclEvent
from .._sycl_queue import SyclQueue, _drain_deferred_releases
from ._seq_order_keeper import _OrderManager


class OrderManagerProtocol(Protocol):
    """Interface of order managers returned by ``SequentialOrderManager``."""

    def add_event_pair(
        self,
        host_task_ev: SyclEvent | Sequence[SyclEvent],
        comp_ev: SyclEvent | Sequence[SyclEvent],
    ) -> None: ...

    @property
    def num_host_task_events(self) -> int: ...

    @property
    def num_submitted_events(self) -> int: ...

    @property
    def host_task_events(self) -> list[SyclEvent]: ...

    @property
    def submitted_events(self) -> list[SyclEvent]: ...

    def wait(self) -> None: ...

    def __copy__(self) -> "OrderManagerProtocol": ...


class _SequentialOrderManager:
    """
    Class to orchestrate default sequential order
    of the tasks offloaded from Python.

    Record offloaded tasks with :meth:`add_event` and use
    :attr:`submitted_events` as the dependencies of the tasks that follow
    them. Record events that gate the release of objects used by a task with
    :meth:`add_cleanup_event`: they are waited on, but never become
    dependencies of later tasks.
    """

    def __init__(self):
        self._state = _OrderManager(16)

    def __del__(self):
        if sys.is_finalizing():
            return
        _local = self._state
        SyclEvent.wait_for(_local.get_submitted_events())
        SyclEvent.wait_for(_local.get_cleanup_events())

    def add_event_pair(self, host_task_ev, comp_ev):
        _local = self._state
        if isinstance(host_task_ev, SyclEvent) and isinstance(
            comp_ev, SyclEvent
        ):
            _local.add_to_both_events(host_task_ev, comp_ev)
        else:
            if not isinstance(host_task_ev, (list, tuple)):
                host_task_ev = (host_task_ev,)
            if not isinstance(comp_ev, (list, tuple)):
                comp_ev = (comp_ev,)
            _local.add_vector_to_both_events(host_task_ev, comp_ev)

    def add_event(self, comp_ev):
        _local = self._state
        if isinstance(comp_ev, SyclEvent):
            _local.add_to_submitted_events(comp_ev)
        else:
            if not isinstance(comp_ev, (list, tuple)):
                comp_ev = (comp_ev,)
            for ev in comp_ev:
                _local.add_to_submitted_events(ev)

    def add_cleanup_event(self, cleanup_ev):
        _local = self._state
        if isinstance(cleanup_ev, SyclEvent):
            _local.add_to_cleanup_events(cleanup_ev)
        else:
            if not isinstance(cleanup_ev, (list, tuple)):
                cleanup_ev = (cleanup_ev,)
            for ev in cleanup_ev:
                _local.add_to_cleanup_events(ev)

    @property
    def num_host_task_events(self):
        return self.num_cleanup_events

    @property
    def num_cleanup_events(self):
        _local = self._state
        return _local.get_num_cleanup_events()

    @property
    def num_submitted_events(self):
        _local = self._state
        return _local.get_num_submitted_events()

    @property
    def host_task_events(self):
        return self.cleanup_events

    @property
    def cleanup_events(self):
        _local = self._state
        return _local.get_cleanup_events()

    @property
    def submitted_events(self):
        _local = self._state
        return _local.get_submitted_events()

    def wait(self):
        _local = self._state
        res = _local.wait()
        # the events are complete, so what was held for those tasks is released
        # here, unless dpctl's thread has already taken it to release; a release
        # runs Python code, which finalization rules out
        if not sys.is_finalizing():
            _drain_deferred_releases()
        return res

    def __copy__(self):
        res = _SequentialOrderManager.__new__(_SequentialOrderManager)
        res._state = _OrderManager(self._state)
        return res


def _as_event_list(ev):
    if isinstance(ev, (list, tuple)):
        return list(ev)
    return [ev]


_use_last_event = SyclQueue.is_get_last_event_available()


class _InOrderManager:
    """
    Order manager for an in-order queue. The queue orders its tasks, so only
    the most recently added events are kept.

    ``submitted_events`` is the event of the last command submitted to the
    queue if :meth:`dpctl.SyclQueue.get_last_event` is available, and the
    most recently added computation events otherwise. ``wait`` waits on the
    queue.
    """

    def __init__(self, q: SyclQueue):
        self._queue = q
        self._host_task_evs = []
        self._comp_evs = []

    def __del__(self):
        if sys.is_finalizing():
            return
        self._queue.wait()

    def add_event_pair(self, host_task_ev, comp_ev):
        self._host_task_evs = _as_event_list(host_task_ev)
        self._comp_evs = _as_event_list(comp_ev)

    @property
    def num_host_task_events(self):
        return len(self._host_task_evs)

    @property
    def num_submitted_events(self):
        return len(self.submitted_events)

    @property
    def host_task_events(self):
        return list(self._host_task_evs)

    @property
    def submitted_events(self):
        if _use_last_event:
            ev = self._queue.get_last_event()
            if ev is not None:
                return [ev]
        return list(self._comp_evs)

    def wait(self):
        self._queue.wait()

    def __copy__(self):
        res = _InOrderManager(self._queue)
        res._host_task_evs = list(self._host_task_evs)
        res._comp_evs = list(self._comp_evs)
        return res


class SyclQueueToOrderManagerMap:
    """
    Utility class used to ensure sequential ordering of offloaded tasks
    when passed to order manager.

    Maintains a thread-local dictionary mapping SyclQueue instances to
    order managers: _InOrderManager for in-order queues,
    _SequentialOrderManager otherwise.
    """

    def __init__(self):
        # each thread gets its own dictionary of order managers
        self._tls = threading.local()

    def _get_map(self):
        """
        Factory method to get or create an order manager map for the
        current thread.
        """
        try:
            return self._tls.order_manager_map
        except AttributeError:
            m = {}
            self._tls.order_manager_map = m
            return m

    def __getitem__(self, q: SyclQueue) -> OrderManagerProtocol:
        """Get order manager for given SyclQueue"""
        if not isinstance(q, SyclQueue):
            raise TypeError(f"Expected `dpctl.SyclQueue`, got {type(q)}")
        _local = self._get_map()
        om = _local.get(q)
        if om is None:
            if q.is_in_order:
                om = _InOrderManager(q)
            else:
                om = _SequentialOrderManager()
            _local[q] = om
        return om

    def clear(self):
        """Clear content of internal dictionary"""
        _local = self._get_map()
        for v in _local.values():
            v.wait()
        _local.clear()


SequentialOrderManager = SyclQueueToOrderManagerMap()


def _callback(som):
    som.clear()


f = weakref.finalize(SequentialOrderManager, _callback, SequentialOrderManager)
