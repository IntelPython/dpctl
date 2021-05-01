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

"""
Demonstrates SYCL USM memory usage in Python using dpctl.memory.
"""

import dpctl.memory as dpmem

# allocate USM-shared byte-buffer
ms = dpmem.MemoryUSMShared(16)

# allocate USM-device byte-buffer
md = dpmem.MemoryUSMDevice(16)

# allocate USM-host byte-buffer
mh = dpmem.MemoryUSMHost(16)

# specify alignment
mda = dpmem.MemoryUSMDevice(128, alignment=16)

# allocate using given queue,
# i.e. on the device and bound to the context stored in the queue
mdq = dpmem.MemoryUSMDevice(256, queue=mda._queue)

# information about device associate with USM buffer
print("Allocation performed on device:")
mda._queue.sycl_device.print_device_info()
