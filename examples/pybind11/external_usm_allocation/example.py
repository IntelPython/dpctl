#                      Data Parallel Control (dpctl)
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

# coding: utf-8

import external_usm_allocation as eua
import numpy as np

import dpctl
import dpctl.memory as dpm

q = dpctl.SyclQueue()
matr = eua.DMatrix(q, 5, 5)

print(matr)
print(matr.__sycl_usm_array_interface__)

blob = dpm.as_usm_memory(matr)

print(blob.get_usm_type())

Xh = np.array(
    [
        [1, 1, 1, 2, 2],
        [1, 0, 1, 2, 2],
        [1, 1, 0, 2, 2],
        [0, 0, 0, 3, -1],
        [0, 0, 0, -1, 5],
    ],
    dtype="d",
)
host_bytes_view = Xh.reshape((-1)).view(np.ubyte)

blob.copy_from_host(host_bytes_view)

print("")
list_of_lists = matr.tolist()
for row in list_of_lists:
    print(row)
