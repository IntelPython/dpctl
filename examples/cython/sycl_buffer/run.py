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

import syclbuffer as sb
import numpy as np

X = np.random.randn(100, 4)

print("Result computed by NumPy")
print(X.sum(axis=0))
print("Result computed by SYCL extension")
print(sb.columnwise_total(X))


print("")
# controlling where to offload
import dpctl

with dpctl.device_context("opencl:gpu"):
    print("Running on: ", dpctl.get_current_queue().get_sycl_device().name)
    print(sb.columnwise_total(X))

with dpctl.device_context("opencl:cpu"):
    print("Running on: ", dpctl.get_current_queue().get_sycl_device().name)
    print(sb.columnwise_total(X))
