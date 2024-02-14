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

import collections.abc
from itertools import chain

from ._sycl_device import SyclDevice, SyclDeviceCreationError
from ._sycl_device_factory import get_devices


def select_device_with_aspects(required_aspects, excluded_aspects=None):
    """Selects the root :class:`dpctl.SyclDevice` that has the highest
    default selector score among devices that have all aspects in the
    `required_aspects` list, and do not have any aspects in `excluded_aspects`
    list.

    The list of SYCL device aspects can be found in SYCL 2020 specs:

    https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:device-aspects

        :Example:
            .. code-block:: python

                import dpctl
                # select a GPU that supports double precision
                dpctl.select_device_with_aspects(['fp64', 'gpu'])
                # select non-custom device with USM shared allocations
                dpctl.select_device_with_aspects(
                    ['usm_shared_allocations'], excluded_aspects=['custom'])
    """
    if excluded_aspects is None:
        excluded_aspects = []
    if isinstance(required_aspects, str):
        required_aspects = [
            required_aspects,
        ]
    if isinstance(excluded_aspects, str):
        excluded_aspects = [
            excluded_aspects,
        ]
    seq = collections.abc.Sequence
    input_types_ok = isinstance(required_aspects, seq) and isinstance(
        excluded_aspects, seq
    )
    if not input_types_ok:
        raise TypeError(
            "Aspects are expected to be Python sequences, "
            "e.g. lists, of strings"
        )
    for asp in chain(required_aspects, excluded_aspects):
        if not isinstance(asp, str):
            raise TypeError("The list objects must be of a string type")
        if not hasattr(SyclDevice, "has_aspect_" + asp):
            raise AttributeError(f"The {asp} aspect is not supported in dpctl")
    devs = get_devices()
    max_score = 0
    selected_dev = None

    for dev in devs:
        aspect_status = all(
            (
                getattr(dev, "has_aspect_" + asp) is True
                for asp in required_aspects
            )
        )
        aspect_status = aspect_status and not (
            any(
                (
                    getattr(dev, "has_aspect_" + asp) is True
                    for asp in excluded_aspects
                )
            )
        )
        if aspect_status and dev.default_selector_score > max_score:
            max_score = dev.default_selector_score
            selected_dev = dev

    if selected_dev is None:
        raise SyclDeviceCreationError(
            f"Requested device is unavailable: "
            f"required_aspects={required_aspects}, "
            f"excluded_aspects={excluded_aspects}"
        )

    return selected_dev
