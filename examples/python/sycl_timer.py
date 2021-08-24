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

import dpctl


class SyclTimer:
    def __init__(self, host_time=timeit.default_timer, time_scale=1):
        self.timer = host_time
        self.time_scale = time_scale

    def __call__(self, queue=None):
        if isinstance(queue, dpctl.SyclQueue):
            if queue.has_enable_profiling:
                self.queue = queue
            else:
                raise ValueError(
                    "The queue does not contain the enable_profiling property"
                )
        else:
            raise ValueError(
                "The passed queue must be <class 'dpctl._sycl_queue.SyclQueue'>"
            )
        return self.__enter__()

    def __enter__(self):
        self.event_start = dpctl.SyclEventRaw(self.queue.submit_barrier())
        self.host_start = self.timer()
        return self

    def __exit__(self, *args):
        self.event_finish = dpctl.SyclEventRaw(self.queue.submit_barrier())
        self.host_finish = self.timer()

    def dt(self):
        self.event_start.wait()
        self.event_finish.wait()
        return (
            (self.host_finish - self.host_start) * self.time_scale,
            (
                self.event_finish.profiling_info_start
                - self.event_start.profiling_info_end
            )
            / 1e9
            * self.time_scale,
        )
