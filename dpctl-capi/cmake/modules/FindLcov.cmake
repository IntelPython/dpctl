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
# CMake find_package() module for lcov.
#
# Example usage:
#
# find_package(Lcov)
#
# If successful, the following variables will be defined:
# LCOV_EXE- The path to lcov executable
# LCOV_FOUND

find_program(LCOV_EXE lcov)
find_program(GENHTML_EXE genhtml)

find_package_handle_standard_args(Lcov DEFAULT_MSG
    LCOV_EXE
    GENHTML_EXE
)
