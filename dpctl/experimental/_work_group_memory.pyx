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

# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True

cdef class WorkGroupMemory:
    """
    WorkGroupMemory(nbytes)
    Python class representing the ``work_group_memory`` class from the 
    Workgroup Memory oneAPI SYCL extension for low-overhead allocation of local
    memory shared by the workitems in a workgroup.

    Args:
        nbytes (int)
            number of bytes to allocate in local memory.
            Expected to be positive.
    """
    def __cinit__(self, Py_ssize_t nbytes):
        self.nbytes = nbytes

    property nbytes:
        """Local memory size in bytes."""
        def __get__(self):
            return self.nbytes


