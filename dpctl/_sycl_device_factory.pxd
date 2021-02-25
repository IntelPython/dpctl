#                      Data Parallel Control (dpCtl)
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

# distutils: language = c++
# cython: language_level=3

""" The file declares several helper functions to create SyclDevice objects
from SYCL standard device_selectors, to get a list of SyclDevices for a
specific backend or device_type.
"""


cpdef select_accelerator_device()
cpdef select_cpu_device()
cpdef select_default_device()
cpdef select_gpu_device()
cpdef select_host_device()
cpdef list get_devices(backend=*, device_type=*)
