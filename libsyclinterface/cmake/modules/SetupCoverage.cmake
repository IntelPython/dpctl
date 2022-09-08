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

function(setup_coverage_generation)
    # check if lcov is available
    find_package(Lcov REQUIRED)
    # check if llvm-cov version 11 is available
    find_package(LLVMCov 11 REQUIRED)
    # check if llvm-profdata is available
    find_package(LLVMProfdata REQUIRED)

    string(CONCAT PROFILE_FLAGS
        "-fprofile-instr-generate "
        "-fcoverage-mapping "
        "-fno-sycl-use-footer "
#        "-save-temps=obj "
    )

    # Add profiling flags
    set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} ${PROFILE_FLAGS}"
        PARENT_SCOPE
    )
endfunction(setup_coverage_generation)
