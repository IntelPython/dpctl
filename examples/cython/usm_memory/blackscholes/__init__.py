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

from ._blackscholes_usm import black_scholes_price, populate_params

__doc__ = """
This is a toy example module illustrating use of SYCL-based code
to operate on NumPy arrays addressing memory allocated by standard
Python memory allocator.
"""
__license__ = "Apache 2.0"

__all__ = [
    "black_scholes_price",
    "populate_params",
]
