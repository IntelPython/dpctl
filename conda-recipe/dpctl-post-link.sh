#!/usr/bin/bash

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

systemwide_icd=/etc/OpenCL/vendors/intel.icd
local_vendors=$PREFIX/etc/OpenCL/vendors
icd_fn=$local_vendors/intel-ocl-gpu.icd

if [[ -f $systemwide_icd && -d $local_vendors && ! -f $icd_fn ]]; then
    ln -s $systemwide_icd $icd_fn
fi
