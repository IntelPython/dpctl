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

""" This file declares the extension types and functions for the Cython API
    implemented in _sycl_*.pyx files.
"""

# distutils: language = c++
# cython: language_level=3

from dpctl._backend cimport *
from dpctl._sycl_context cimport *
from dpctl._sycl_device cimport *
from dpctl._sycl_device_factory cimport *
from dpctl._sycl_event cimport *
from dpctl._sycl_platform cimport *
from dpctl._sycl_queue cimport *
