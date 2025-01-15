#                       Data Parallel Control (dpctl)
#
#  Copyright 2020-2025 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from .._sycl_device import SyclDevice
from ._usmarray import DLDeviceType


def dldevice_to_sycl_device(dl_dev: tuple):
    if isinstance(dl_dev, tuple):
        if len(dl_dev) != 2:
            raise ValueError("dldevice tuple must have length 2")
    else:
        raise TypeError(
            f"dl_dev is expected to be a 2-tuple, got " f"{type(dl_dev)}"
        )
    if dl_dev[0] != DLDeviceType.kDLOneAPI:
        raise ValueError("dldevice type must be kDLOneAPI")
    return SyclDevice(str(dl_dev[1]))


def sycl_device_to_dldevice(dev: SyclDevice):
    if not isinstance(dev, SyclDevice):
        raise TypeError(
            "dev is expected to be a SyclDevice, got " f"{type(dev)}"
        )
    return (DLDeviceType.kDLOneAPI, dev.get_device_id())
