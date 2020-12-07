# ===-------- tests/dpctl_tests/__init__.py - dpctl  ------*- Python -*-----===#
#
#                      Data Parallel Control (dpCtl)
#
# Copyright 2020 Intel Corporation
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
#
# ===-----------------------------------------------------------------------===#
#
# \file
# Top-level module of all dpctl Python unit test cases.
# ===-----------------------------------------------------------------------===#

from .test_dparray import *
from .test_dump_functions import *
from .test_sycl_device import *
from .test_sycl_kernel_submit import *
from .test_sycl_program import *
from .test_sycl_queue import *
from .test_sycl_queue_manager import *
from .test_sycl_queue_memcpy import *
from .test_sycl_usm import *
