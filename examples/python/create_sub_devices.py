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

"""Demonstrates how to create sub-devices on a device.
"""

import dpctl


# For example, let's take a cpu device
d = dpctl.SyclDevice("cpu")

# Create sub-devices equally. The returned vector contains
# as many sub devices as can be created such that each sub-device
# contains count compute units.
d.create_sub_devices(partition=2)

# Create sub-devices by counts. Returns a vector of sub devices
# partitioned from this SYCL device based on the counts parameter.
d.create_sub_devices(partition=[2, 2])

# Create sub-devices by affinity domain based on the domain
# parameter.
d.create_sub_devices(partition="numa")
