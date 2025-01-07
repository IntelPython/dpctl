#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import timeit

from . import SyclQueue

__doc__ = "This module implements :class:`dpctl.SyclTimer`."


class HostDeviceDuration:
    def __init__(self, host_dt, device_dt):
        self._host_dt = host_dt
        self._device_dt = device_dt

    def __repr__(self):
        return f"(host_dt={self._host_dt}, device_dt={self._device_dt})"

    def __str__(self):
        return f"(host_dt={self._host_dt}, device_dt={self._device_dt})"

    def __iter__(self):
        yield from [self._host_dt, self._device_dt]

    @property
    def host_dt(self):
        return self._host_dt

    @property
    def device_dt(self):
        return self._device_dt


class BaseDeviceTimer:
    __slots__ = ["queue"]

    def __init__(self, sycl_queue):
        if not isinstance(sycl_queue, SyclQueue):
            raise TypeError(f"Expected type SyclQueue, got {type(sycl_queue)}")
        self.queue = sycl_queue


class QueueBarrierDeviceTimer(BaseDeviceTimer):
    __slots__ = []

    def __init__(self, sycl_queue):
        super(QueueBarrierDeviceTimer, self).__init__(sycl_queue)

    def get_event(self):
        return self.queue.submit_barrier()


class OrderManagerDeviceTimer(BaseDeviceTimer):
    __slots__ = ["_order_manager", "_submit_empty_task_fn"]

    def __init__(self, sycl_queue):
        import dpctl.utils._seq_order_keeper as s_ok
        from dpctl.utils import SequentialOrderManager as seq_om

        super(OrderManagerDeviceTimer, self).__init__(sycl_queue)
        self._order_manager = seq_om[self.queue]
        self._submit_empty_task_fn = s_ok._submit_empty_task

    def get_event(self):
        ev = self._submit_empty_task_fn(
            sycl_queue=self.queue, depends=self._order_manager.submitted_events
        )
        self._order_manager.add_event_pair(ev, ev)
        return ev


class SyclTimer:
    """
    Context to time execution of tasks submitted to :class:`dpctl.SyclQueue`.

    :Example:
        .. code-block:: python

            import dpctl

            # Create a default SyclQueue
            q = dpctl.SyclQueue(property="enable_profiling")

            # create the timer
            milliseconds_sc = 1e3
            timer = dpctl.SyclTimer(time_scale = milliseconds_sc)

            untimed_code_block_1
            # use the timer
            with timer(queue=q):
                timed_code_block1

            untimed_code_block_2

            # use the timer
            with timer(queue=q):
                timed_code_block2

            untimed_code_block_3

            # retrieve elapsed times in milliseconds
            wall_dt, device_dt = timer.dt

    .. note::
        The timer submits tasks to the queue at the entrance and the
        exit of the context and uses profiling information from events
        associated with these submissions to perform the timing. Thus
        :class:`dpctl.SyclTimer` requires the queue with ``"enable_profiling"``
        property. In order to be able to collect the profiling information,
        the ``dt`` property ensures that both tasks submitted by the timer
        complete their execution and thus effectively synchronizes the queue.

        Execution of the above example results in the following task graph,
        where each group of tasks is ordered after the one preceding it,
        ``[tasks_of_untimed_block1]``, ``[timer_fence_start_task]``,
        ``[tasks_of_timed_block1]``, ``[timer_fence_finish_task]``,
        ``[tasks_of_untimed_block2]``, ``[timer_fence_start_task]``,
        ``[tasks_of_timed_block2]``, ``[timer_fence_finish_task]``,
        ``[tasks_of_untimed_block3]``.

        ``device_timer`` keyword argument controls the type of tasks submitted.
        With ``"queue_barrier"`` value, queue barrier tasks are used. With
        ``"order_manager"`` value, a single empty body task is inserted
        and order manager (used by all `dpctl.tensor` operations) is used to
        order these tasks so that they fence operations performed within
        timer's context.

        Timing offloading operations that do not use the order manager with
        the timer that uses ``"order_manager"`` as ``device_timer`` value
        will be misleading becaused the tasks submitted by the timer will not
        be ordered with respect to tasks we intend to time.

        Note, that host timer effectively measures the time of task
        submissions. To measure host timer wall-time that includes execution
        of submitted tasks, make sure to include synchronization point in
        the timed block.

        :Example:
            .. code-block:: python

            with timer(q):
                timed_block
                q.wait()

    Args:
        host_timer (callable, optional):
            A callable such that host_timer() returns current
            host time in seconds.
            Default: :py:func:`timeit.default_timer`.
        device_timer (Literal["queue_barrier", "order_manager"], optional):
            Device timing method. Default: "queue_barrier".
        time_scale (Union[int, float], optional):
            Ratio of one second and the unit of time-scale of interest.
            Default: ``1``.
    """

    def __init__(
        self, host_timer=timeit.default_timer, device_timer=None, time_scale=1
    ):
        """
        Create new instance of :class:`.SyclTimer`.

        Args:
            host_timer (callable, optional)
                A function that takes no arguments and returns a value
                measuring time.
                Default: :meth:`timeit.default_timer`.
            device_timer (Literal["queue_barrier", "order_manager"], optional):
                Device timing method. Default: "queue_barrier"
            time_scale (Union[int, float], optional):
                Scaling factor applied to durations measured by
                the host_timer. Default: ``1``.
        """
        self.timer = host_timer
        self.time_scale = time_scale
        self.queue = None
        self.host_times = []
        self.bracketing_events = []
        self._context_data = list()
        if device_timer is None:
            device_timer = "queue_barrier"
        if device_timer == "queue_barrier":
            self._device_timer_class = QueueBarrierDeviceTimer
        elif device_timer == "order_manager":
            self._device_timer_class = OrderManagerDeviceTimer
        else:
            raise ValueError(
                "Supported values for device_timer keyword are "
                "'queue_barrier', 'order_manager', got "
                f"'{device_timer}'"
            )
        self._device_timer = None

    def __call__(self, queue=None):
        if isinstance(queue, SyclQueue):
            if queue.has_enable_profiling:
                self.queue = queue
                self._device_timer = self._device_timer_class(queue)
            else:
                raise ValueError(
                    "The given queue was not created with the "
                    "enable_profiling property"
                )
        else:
            raise TypeError(
                "The passed queue must have type dpctl.SyclQueue, "
                f"got {type(queue)}"
            )
        return self

    def __enter__(self):
        _event_start = self._device_timer.get_event()
        _host_start = self.timer()
        self._context_data.append((_event_start, _host_start))
        return self

    def __exit__(self, *args):
        _event_end = self._device_timer.get_event()
        _host_end = self.timer()
        _event_start, _host_start = self._context_data.pop()
        self.host_times.append((_host_start, _host_end))
        self.bracketing_events.append((_event_start, _event_end))

    @property
    def dt(self):
        """Returns a pair of elapsed times ``host_dt`` and
        ``device_dt``.

        The ``host_dt`` is the duration as measured by the host
        timer, while the ``device_dt`` is the duration as measured by
        the device timer and encoded in profiling events.

        Returns:
            HostDeviceDuration:
                Data class with ``host_dt`` and ``device_dt`` members which
                supports unpacking into a 2-tuple.

        :Example:

            .. code-block:: python

                import dpctl
                from dpctl import tensor

                q = dpctl.SyclQueue(property="enable_profiling")

                device = tensor.Device.create_device(q)
                timer = dpctl.SyclTimer()

                with timer(q):
                    x = tensor.linspace(-4, 4, num=10**6, dtype="float32")
                    e = tensor.exp(-0.5 * tensor.square(x))
                    s = tensor.sin(2.3 * x + 0.11)
                    f = e * s

                host_dt, device_dt = timer.dt

        .. note::
            Since different timers are used to measure host and device
            durations, one should not expect that ``host_dt`` is always
            strictly greater than ``device_dt``.

            Use tracing tools like ``onetrace``, or ``unitrace`` from
            `intel/pti-gpu <https://github.com/intel/pti-gpu>`_ repository
            for more accurate measurements.
        """
        for es, ef in self.bracketing_events:
            es.wait()
            ef.wait()
        host_dt = sum(tf - ts for ts, tf in self.host_times) * self.time_scale
        dev_dt = sum(
            ef.profiling_info_start - es.profiling_info_end
            for es, ef in self.bracketing_events
        ) * (1e-9 * self.time_scale)
        return HostDeviceDuration(host_dt, dev_dt)
