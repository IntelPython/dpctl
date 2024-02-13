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

# cython: language=c++
# cython: language_level=3

cimport dpctl as c_dpctl

import dpctl


def call_create_from_context_and_devices():
    cdef c_dpctl.SyclQueue q
    d = dpctl.SyclDevice()
    ctx = dpctl.SyclContext(d)
    # calling static method
    q = c_dpctl.SyclQueue._create_from_context_and_device(
        <c_dpctl.SyclContext> ctx,
        <c_dpctl.SyclDevice> d
    )
    return q
