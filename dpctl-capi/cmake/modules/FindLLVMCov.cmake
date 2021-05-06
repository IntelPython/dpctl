#                       Data Parallel Control (dpctl)
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
#
# CMake find_package() module for llvm-cov.
#
# Example usage:
#
# find_package(LLVMCov)
#
# If successful, the following variables will be defined:
# LLVMCov_EXE- The path to llvm-cov executable
# LLVMCov_FOUND
# LLVMCov_VERSION

find_program(LLVMCov_EXE llvm-cov)

# get the version of llvm-cov
execute_process(
    COMMAND ${LLVMCov_EXE} "--version"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE version_string
)

# If llvm-cov is found then set the package variables
if(${result} MATCHES "0")
    string(REPLACE "\n" ";" VERSION_LIST "${version_string}")
    list(GET VERSION_LIST 1 ver_line)
    # Extract the llvm-cov version
    string(REGEX MATCH "[0-9]+\.[0-9]+\.[0-9]+" LLVMCov_VERSION ${ver_line})
    # Split out the version into major, minor an patch
    string(REPLACE "." ";" VERSION_LIST1 "${LLVMCov_VERSION}")
    list(GET VERSION_LIST1 0 LLVMCov_VERSION_MAJOR)
    list(GET VERSION_LIST1 1 LLVMCov_VERSION_MINOR)
    list(GET VERSION_LIST1 2 LLVMCov_VERSION_PATCH)
endif()

# Check if a specific version of llvm-cov is required.
if(LLVMCov_FIND_VERSION AND (DEFINED LLVMCov_VERSION))
    string(COMPARE LESS_EQUAL
        ${LLVMCov_FIND_VERSION_MAJOR}
        ${LLVMCov_VERSION_MAJOR}
        VERSION_MATCH
    )
    if(VERSION_MATCH)
        set(LLVMCov_FOUND TRUE)
    endif()
else()
    set(LLVMCov_FOUND TRUE)
endif()


find_package_handle_standard_args(LLVMCov DEFAULT_MSG
    LLVMCov_EXE
    LLVMCov_FOUND
    LLVMCov_VERSION
)
