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

"""
A collection of utility functions.
"""

from ._intel_device_info import intel_device_info
from ._onetrace_context import onetrace_enabled
from ._order_manager import SequentialOrderManager

__all__ = [
    "onetrace_enabled",
    "intel_device_info",
    "SequentialOrderManager",
]
