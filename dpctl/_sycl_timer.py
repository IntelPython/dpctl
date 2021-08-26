#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2021 Intel Corporation
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


class SyclTimer:
    """
    SyclTimer(host_timer=timeit.default_timer, time_scale=1)
    Python class to measure device time of execution of commands submitted to
    :class:`dpctl.SyclQueue` as well as the wall-time.

    :Example:
        .. code-block:: python

            import dpctl

            # Create a default SyclQueue
            q = dpctl.SyclQueue(property='enable_profiling')

            # create the timer
            miliseconds_sc = 1e-3
            timer = dpctl.SyclTimer(time_scale = miliseconds_sc)

            # use the timer
            with timer(queue=q):
                code_block

            # retrieve elapsed times in miliseconds
            sycl_dt, wall_dt = timer.dt

    Remark:
       The timer synchronizes the queue at the entrance and the
       exit of the context.

    Args:
       host_timer (callable): A callable such that host_timer() returns current
       host time in seconds.
       time_scale (int, float): Ratio of the unit of time of interest and
       one second.
    """

    def __init__(self, host_timer=timeit.default_timer, time_scale=1):
        self.timer = host_timer
        self.time_scale = time_scale
        self.queue = None

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
                "got {}".format(type(queue))
            )
        return self

    def __enter__(self):
        self.event_start = self.queue.submit_barrier()
        self.host_start = self.timer()
        return self

    def __exit__(self, *args):
        self.event_finish = self.queue.submit_barrier()
        self.host_finish = self.timer()

    @property
    def dt(self):
        self.event_start.wait()
        self.event_finish.wait()
        return (
            (self.host_finish - self.host_start) * self.time_scale,
            (
                self.event_finish.profiling_info_start
                - self.event_start.profiling_info_end
            )
            * (1e-9 * self.time_scale),
        )
