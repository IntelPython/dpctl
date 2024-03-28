#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2024 Intel Corporation
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


class SyclTimer:
    """
    Context to measure device time and host wall-time of execution
    of commands submitted to :class:`dpctl.SyclQueue`.

    :Example:
        .. code-block:: python

            import dpctl

            # Create a default SyclQueue
            q = dpctl.SyclQueue(property='enable_profiling')

            # create the timer
            milliseconds_sc = 1e-3
            timer = dpctl.SyclTimer(time_scale = milliseconds_sc)

            # use the timer
            with timer(queue=q):
                code_block

            # retrieve elapsed times in milliseconds
            wall_dt, device_dt = timer.dt

    Note:
        The timer submits barriers to the queue at the entrance and the
        exit of the context and uses profiling information from events
        associated with these submissions to perform the timing. Thus
        :class:`dpctl.SyclTimer` requires the queue with "enable_profiling"
        property. In order to be able to collect the profiling information,
        the `dt` property ensures that both submitted barriers complete their
        execution and thus effectively synchronizes the queue.

    Args:
        host_timer (callable, optional):
            A callable such that host_timer() returns current
            host time in seconds.
            Default: :py:func:`timeit.default_timer`.
        time_scale (Union[int, float], optional):
            Ratio of the unit of time of interest and one second.
            Default: `1`.
    """

    def __init__(self, host_timer=timeit.default_timer, time_scale=1):
        self.timer = host_timer
        self.time_scale = time_scale
        self.queue = None
        self.host_times = []
        self.bracketing_events = []

    def __call__(self, queue=None):
        if isinstance(queue, SyclQueue):
            if queue.has_enable_profiling:
                self.queue = queue
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
        self._event_start = self.queue.submit_barrier()
        self._host_start = self.timer()
        return self

    def __exit__(self, *args):
        self.host_times.append((self._host_start, self.timer()))
        self.bracketing_events.append(
            (self._event_start, self.queue.submit_barrier())
        )
        del self._event_start
        del self._host_start

    @property
    def dt(self):
        """Returns a pair of elapsed times (host_dt, device_dt).

        The host_dt is the duration as measured by the host
        timer, while the device_dt is the duration as measured by
        the device timer and encoded in profiling events."""
        for es, ef in self.bracketing_events:
            es.wait()
            ef.wait()
        host_dt = sum(tf - ts for ts, tf in self.host_times) * self.time_scale
        dev_dt = sum(
            ef.profiling_info_start - es.profiling_info_end
            for es, ef in self.bracketing_events
        ) * (1e-9 * self.time_scale)
        return HostDeviceDuration(host_dt, dev_dt)
