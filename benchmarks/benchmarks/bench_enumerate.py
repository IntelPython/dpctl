#                      Data Parallel Control (dpctl)
#
# Copyright 2026 Intel Corporation
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

"""Benchmarks for device enumeration and device selection."""

from asv_runner.benchmarks.mark import SkipNotImplemented

import dpctl

_DEVICE_TYPES = ["all", "cpu", "gpu"]
_BACKENDS = ["all", "opencl", "level_zero"]


class Enumerate:
    """get_devices / get_num_devices across device_type and backend."""

    params = [_DEVICE_TYPES, _BACKENDS]
    param_names = ["device_type", "backend"]

    def setup(self, device_type, backend):
        # first enumeration initializes the SYCL platform list
        dpctl.get_devices(backend=backend, device_type=device_type)

    def time_get_devices(self, device_type, backend):
        dpctl.get_devices(backend=backend, device_type=device_type)

    def time_get_num_devices(self, device_type, backend):
        dpctl.get_num_devices(backend=backend, device_type=device_type)


class Select:
    """Device selectors and availability predicates."""

    def setup(self):
        dpctl.select_default_device()
        try:
            dpctl.select_device_with_aspects("fp64")
            self.has_fp64 = True
        except dpctl.SyclDeviceCreationError:
            self.has_fp64 = False

    def time_select_default_device(self):
        dpctl.select_default_device()

    def time_select_cpu_device(self):
        if dpctl.has_cpu_devices():
            dpctl.select_cpu_device()

    def time_select_gpu_device(self):
        if dpctl.has_gpu_devices():
            dpctl.select_gpu_device()

    def time_has_cpu_devices(self):
        dpctl.has_cpu_devices()

    def time_has_gpu_devices(self):
        dpctl.has_gpu_devices()

    def time_select_device_with_aspects(self):
        if not self.has_fp64:
            raise SkipNotImplemented("no device with the fp64 aspect")
        dpctl.select_device_with_aspects("fp64")
