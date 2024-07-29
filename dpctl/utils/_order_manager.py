import weakref
from collections import defaultdict
from contextvars import ContextVar

from .._sycl_event import SyclEvent
from .._sycl_queue import SyclQueue
from ._seq_order_keeper import _OrderManager


class _SequentialOrderManager:
    """
    Class to orchestrate default sequential order
    of the tasks offloaded from Python.
    """

    def __init__(self):
        self._state = _OrderManager(16)

    def __dealloc__(self):
        _local = self._state
        SyclEvent.wait_for(_local.get_submitted_events())
        SyclEvent.wait_for(_local.get_host_task_events())

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

    @property
    def num_host_task_events(self):
        _local = self._state
        return _local.get_num_host_task_events()

    @property
    def num_submitted_events(self):
        _local = self._state
        return _local.get_num_submitted_events()

    @property
    def host_task_events(self):
        _local = self._state
        return _local.get_host_task_events()

    @property
    def submitted_events(self):
        _local = self._state
        return _local.get_submitted_events()

    def wait(self):
        _local = self._state
        return _local.wait()

    def __copy__(self):
        res = _SequentialOrderManager.__new__(_SequentialOrderManager)
        res._state = _OrderManager(self._state)
        return res


class SyclQueueToOrderManagerMap:
    """Utility class to ensure sequential ordering of offloaded
    tasks issued by dpctl.tensor functions"""

    def __init__(self):
        self._map = ContextVar(
            "global_order_manager_map",
            default=defaultdict(_SequentialOrderManager),
        )

    def __getitem__(self, q: SyclQueue) -> _SequentialOrderManager:
        """Get order manager for given SyclQueue"""
        _local = self._map.get()
        if not isinstance(q, SyclQueue):
            raise TypeError(f"Expected `dpctl.SyclQueue`, got {type(q)}")
        if q in _local:
            return _local[q]
        else:
            v = _local[q]
            _local[q] = v
            return v

    def clear(self):
        """Clear content of internal dictionary"""
        _local = self._map.get()
        for v in _local.values():
            v.wait()
        _local.clear()


SequentialOrderManager = SyclQueueToOrderManagerMap()


def _callback(som):
    som.clear()


f = weakref.finalize(SequentialOrderManager, _callback, SequentialOrderManager)
