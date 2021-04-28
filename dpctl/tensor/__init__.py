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

"""
    **Data Parallel Tensor Collection**

    `dpctl.tensor` is an experimental collection of tensor implementations
    that will implement future Python data API (https://data-apis.github.io/array-api/latest/).

    Available tensor implementations:

    * `numpy_usm_shared`: Provides a `numpy.ndarray` sub-class whose \
    underlying memory buffer is allocated with a USM shared memory allocator.

"""

import dpctl.tensor.numpy_usm_shared
