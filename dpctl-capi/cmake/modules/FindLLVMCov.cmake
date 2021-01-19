#               Data Parallel Control Library (dpCtl)
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
# CMake find_package() module for llvm-cov.
#
# Example usage:
#
# find_package(LLVMCov)
#
# If successful, the following variables will be defined:
# LLVMCov_BIN - Path to llvm-cov
# LLVMProfdata_BIN - Path to llvm-profdata
# LLVMCov_FOUND

find_program(LLVMCov_BIN llvm-cov)
find_program(LLVMProfdata_BIN llvm-profdata)

find_package_handle_standard_args(LLVMCov DEFAULT_MSG
    LLVMCov_BIN
    LLVMProfdata_BIN
)
