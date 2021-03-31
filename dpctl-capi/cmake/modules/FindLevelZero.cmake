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
# CMake find_package() module for the Level Zero loader library and headers.
#
# Example usage:
#
# find_package(LevelZero)
#
# If successful, the following variables will be defined:
# LEVEL_ZERO_INCLUDE_DIR
# LEVEL_ZERO_LIBRARY - the full path to the ze_loader library
# TODO: Add a way to record the version of the level_zero library

find_library(LEVEL_ZERO_LIBRARY ze_loader HINTS $ENV{L0_LIB_DIR})
find_path(LEVEL_ZERO_INCLUDE_DIR NAMES level_zero/zet_api.h HINTS $ENV{L0_INCLUDE_DIR})

find_package_handle_standard_args(LevelZero DEFAULT_MSG
    LEVEL_ZERO_INCLUDE_DIR
    LEVEL_ZERO_LIBRARY
)
