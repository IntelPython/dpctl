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

    def add_event(self, comp_ev: SyclEvent | Sequence[SyclEvent]) -> None: ...

    def add_cleanup_event(
        self, cleanup_ev: SyclEvent | Sequence[SyclEvent]
    ) -> None: ...

    @property
    def num_host_task_events(self) -> int: ...

    @property
    def num_cleanup_events(self) -> int: ...

    @property
    def num_submitted_events(self) -> int: ...

    @property
    def host_task_events(self) -> list[SyclEvent]: ...

    @property
    def cleanup_events(self) -> list[SyclEvent]: ...

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

    def __init__(self, q=None):
        self._state = _OrderManager(16)
        # the queue whose tasks are recorded, a copy of the one it is looked up
        # by, so that the map never references the queue that is its key
        self._queue = q
        # weak reference to the queue the map has it under, None if that
        # queue is gone
        self._key = None
        # set while the map does not hold it, once its work was complete
        self._dropped = False

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

    def _is_idle(self):
        """
        Whether all the work recorded here is complete; never blocks.

        For an in-order queue, work submitted without being recorded needs no
        order manager: the queue keeps its order, and the last event of the
        queue is a dependency of the work submitted after it. That event is not
        looked at, as Level Zero v2 reports it submitted until waited on.
        """
        _local = self._state
        # both prune the complete events first
        return not (
            _local.get_submitted_events() or _local.get_cleanup_events()
        )

    def __copy__(self):
        res = _SequentialOrderManager.__new__(_SequentialOrderManager)
        res._state = _OrderManager(self._state)
        res._queue = self._queue
        res._key = None
        res._dropped = False
        return res


_use_last_event = SyclQueue.is_get_last_event_available()
_weak_ref = weakref.ref


class _InOrderManager(_SequentialOrderManager):
    """
    Order manager for an in-order queue.

    Holds the added events which are not complete, as
    :class:`_SequentialOrderManager` does. ``submitted_events`` also has the
    event of the last command submitted to the queue if
    :meth:`dpctl.SyclQueue.get_last_event` is available. ``wait`` waits on
    the queue.
    """

    def __del__(self):
        if sys.is_finalizing():
            return
        # the queue keeps its own order, so one the map dropped need not wait
        # for work submitted to the queue elsewhere since
        if not self._dropped:
            self._queue.wait()
        super().__del__()

    @property
    def num_submitted_events(self):
        return len(self.submitted_events)

    @property
    def submitted_events(self):
        evs = self._state.get_submitted_events()
        if _use_last_event:
            ev = self._queue.get_last_event()
            if ev is not None:
                evs.append(ev)
        return evs

    def wait(self):
        self._queue.wait()
        return super().wait()

    def __copy__(self):
        res = _InOrderManager.__new__(_InOrderManager)
        res._state = _OrderManager(self._state)
        res._queue = self._queue
        res._key = None
        res._dropped = False
        return res


class _QueueDeath(weakref.ref):
    """
    Weak reference to a queue used as a key of the map, which tells the map
    that the queue is gone. ``key`` is the weak reference the map is keyed by.
    """

    __slots__ = ("key",)

    def __new__(cls, q, callback, key):
        return super().__new__(cls, q, callback)

    def __init__(self, q, callback, key):
        super().__init__(q, callback)
        self.key = key


class _ThreadMap:
    """The order managers of one thread."""

    __slots__ = ("managers", "deaths", "dead", "owned", "retired", "dropped")

    def __init__(self):
        # weakref.ref(queue) -> order manager, for the queues alive
        self.managers = {}
        # weakref.ref(queue) -> _QueueDeath of the queue
        self.deaths = {}
        # _QueueDeath of the keys that are gone, appended by the weak
        # reference callback on whichever thread drops the queue
        self.dead = []
        # copy of the queue -> order manager, for all the map holds: those of
        # the queues alive and those of the queues gone with work not complete
        self.owned = {}
        # id -> order manager of a queue that is gone, with work not complete
        self.retired = {}
        # copy of the queue -> order manager of a queue that is gone, which
        # the map dropped once its work was complete, for as long as anything
        # else holds it
        self.dropped = weakref.WeakValueDictionary()


class SyclQueueToOrderManagerMap:
    """
    Utility class used to ensure sequential ordering of offloaded tasks
    when passed to order manager.

    Maintains a thread-local dictionary mapping SyclQueue instances to
    order managers: _InOrderManager for in-order queues,
    _SequentialOrderManager otherwise.

    Queues are held by weak reference, and a ``sycl::queue`` has one order
    manager in a thread for as long as the map or anything else holds it: the
    map does while the queue it was looked up by is alive and while its work
    is not complete. A queue looked up after that queue is gone gets the same
    order manager if it is still held, so that the work recorded in it stays
    ordered. Nothing waits when a queue is gone, as that can be on any thread,
    including one that a recorded task is waiting on.
    """

    def __init__(self):
        # each thread gets its own dictionary of order managers
        self._tls = threading.local()

    def _get_thread_map(self):
        """
        Factory method to get or create an order manager map for the
        current thread.
        """
        try:
            return self._tls.thread_map
        except AttributeError:
            m = _ThreadMap()
            self._tls.thread_map = m
            return m

    @staticmethod
    def _prune(m):
        """
        Drop the order managers of the queues that are gone once their work
        is complete. Runs on the thread that owns ``m``.
        """
        dead = m.dead
        while dead:
            death = dead.pop()
            key = death.key
            if m.deaths.get(key) is not death:
                # taken over by another queue
                continue
            del m.deaths[key]
            om = m.managers.pop(key)
            om._key = None
            m.retired[id(om)] = om
        for k, om in list(m.retired.items()):
            if om._is_idle():
                del m.retired[k]
                if m.owned.get(om._queue) is om:
                    del m.owned[om._queue]
                om._dropped = True
                m.dropped[om._queue] = om

    @staticmethod
    def _add(m, q):
        """Give ``q`` an order manager, as ``m`` has none under it."""
        # one the sycl::queue already has: its queue is gone, even if the map
        # is yet to be told
        om = m.owned.get(q)
        if om is None:
            om = m.dropped.pop(q, None)
            if om is not None:
                om._dropped = False
                m.owned[om._queue] = om
        if om is not None:
            old = om._key
            if old is not None:
                m.managers.pop(old, None)
                m.deaths.pop(old, None)
            m.retired.pop(id(om), None)
        else:
            # the same sycl::queue in a new object, so that the value does
            # not keep its key alive
            q_copy = SyclQueue(q._get_capsule())
            if q.is_in_order:
                om = _InOrderManager(q_copy)
            else:
                om = _SequentialOrderManager(q_copy)
            m.owned[q_copy] = om
        key = _weak_ref(q)
        om._key = key
        m.managers[key] = om
        m.deaths[key] = _QueueDeath(q, m.dead.append, key)
        return om

    def __getitem__(self, q: SyclQueue) -> OrderManagerProtocol:
        """Get order manager for given SyclQueue"""
        if not isinstance(q, SyclQueue):
            raise TypeError(f"Expected `dpctl.SyclQueue`, got {type(q)}")
        try:
            m = self._tls.thread_map
        except AttributeError:
            m = self._get_thread_map()
        if m.dead or m.retired:
            self._prune(m)
        # the key the map holds, so the lookup finds it by identity
        om = m.managers.get(_weak_ref(q))
        if om is None:
            om = self._add(m, q)
        return om

    def clear(self):
        """Clear content of internal dictionary"""
        m = self._get_thread_map()
        for om in list(m.owned.values()):
            om.wait()
        for om in list(m.dropped.values()):
            om.wait()
        m.managers.clear()
        m.deaths.clear()
        m.dead.clear()
        m.owned.clear()
        m.retired.clear()
        m.dropped.clear()


SequentialOrderManager = SyclQueueToOrderManagerMap()


def _callback(som):
    som.clear()


f = weakref.finalize(SequentialOrderManager, _callback, SequentialOrderManager)
