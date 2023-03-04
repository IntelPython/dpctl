#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2022 Intel Corporation
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

from ._compute_follows_data import (
    ExecutionPlacementError,
    get_coerced_usm_type,
    get_execution_queue,
    validate_usm_type,
)
from ._onetrace_context import onetrace_enabled

__all__ = [
    "get_execution_queue",
    "get_coerced_usm_type",
    "validate_usm_type",
    "onetrace_enabled",
    "ExecutionPlacementError",
]
