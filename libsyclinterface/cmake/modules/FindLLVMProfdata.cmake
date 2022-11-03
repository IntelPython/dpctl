#                       Data Parallel Control (dpctl)
#
# Copyright 2020-2022 Intel Corporation
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
# LLVMProfdata_EXE The path to llvm-cov executable
# LLVMProfdata_FOUND

if(DEFINED ENV{LLVM_TOOLS_HOME})
    find_program(LLVMProfdata_EXE
        llvm-profdata
        PATHS $ENV{LLVM_TOOLS_HOME}
        NO_DEFAULT_PATH
    )
    if(${LLVMProfdata_EXE} STREQUAL "LLVMProfdata_EXE-NOTFOUND")
        message(WARN
            "$ENV{LLVM_TOOLS_HOME} does not have an llvm-profdata executable"
        )
    endif()
else()
    find_program(LLVMProfdata_EXE llvm-profdata)
endif()

find_package_handle_standard_args(LLVMProfdata DEFAULT_MSG
    LLVMProfdata_EXE
)
