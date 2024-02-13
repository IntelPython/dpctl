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

"""Helper module for dpctl/tests
"""

from ._helper import (
    create_invalid_capsule,
    get_queue_or_skip,
    has_cpu,
    has_gpu,
    has_sycl_platforms,
    skip_if_dtype_not_supported,
)

__all__ = [
    "create_invalid_capsule",
    "has_cpu",
    "has_gpu",
    "has_sycl_platforms",
    "get_queue_or_skip",
    "skip_if_dtype_not_supported",
]
