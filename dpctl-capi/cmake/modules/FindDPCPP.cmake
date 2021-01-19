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
# CMake find_package() module for the DPCPP compiler and development
# environment.
#
# Example usage:
#
# find_package(DPCPP)
#
# If successful, the following variables will be defined:
# DPCPP_FOUND
# DPCPP_VERSION
# DPCPP_INCLUDE_DIR
# DPCPP_SYCL_INCLUDE_DIR
# DPCPP_LIBRARY_DIR
# DPCPP_SYCL_LIBRARY
# DPCPP_OPENCL_LIBRARY

include( FindPackageHandleStandardArgs )

string(COMPARE EQUAL "${DPCPP_INSTALL_DIR}" "" no_dpcpp_root)
if(${no_dpcpp_root})
    message(STATUS "Set the DPCPP_ROOT argument providing the path to \
                         a dpcpp installation.")
    return()
endif()

if(WIN32 OR UNIX)
    set(dpcpp_cmd "${DPCPP_INSTALL_DIR}/bin/dpcpp")
    set(dpcpp_arg "--version")
else()
    message(FATAL_ERROR "Unsupported system.")
endif()

# Check if dpcpp is available
execute_process(
    COMMAND ${dpcpp_cmd} ${dpcpp_arg}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    RESULT_VARIABLE dpcpp_result
    OUTPUT_VARIABLE dpcpp_ver
)

# If dpcpp is found then set then set the package variables
if(${dpcpp_result} MATCHES "0")
    string(REPLACE "\n" ";" DPCPP_VERSION_LIST "${dpcpp_ver}")
    list(GET DPCPP_VERSION_LIST 0 dpcpp_ver_line)
    foreach(X ${DPCPP_VERSION_LIST})
        message(STATUS "dpcpp ver[${dpcpp_result}]: ${X}")
    endforeach()

    # set package-level variables
    set(DPCPP_ROOT ${DPCPP_INSTALL_DIR})
    list(POP_FRONT DPCPP_VERSION_LIST DPCPP_VERSION)
    set(DPCPP_INCLUDE_DIR ${DPCPP_INSTALL_DIR}/include)
    set(DPCPP_SYCL_INCLUDE_DIR ${DPCPP_INSTALL_DIR}/include/sycl)
    set(DPCPP_LIBRARY_DIR ${DPCPP_INSTALL_DIR}/lib)
    if(WIN32)
        set(DPCPP_SYCL_LIBRARY ${DPCPP_INSTALL_DIR}/lib/sycl.lib)
        set(DPCPP_OPENCL_LIBRARY ${DPCPP_INSTALL_DIR}/lib/OpenCL.lib)
    elseif(UNIX)
        set(DPCPP_SYCL_LIBRARY ${DPCPP_INSTALL_DIR}/lib/libsycl.so)
        set(DPCPP_OPENCL_LIBRARY ${DPCPP_INSTALL_DIR}/lib/libOpenCL.so)
    endif()
else()
    message(STATUS "DPCPP needed to build dpctl_sycl_interface")
    return()
endif()

find_package_handle_standard_args(DPCPP DEFAULT_MSG
    DPCPP_VERSION
    DPCPP_INCLUDE_DIR
    DPCPP_SYCL_INCLUDE_DIR
    DPCPP_LIBRARY_DIR
    DPCPP_SYCL_LIBRARY
    DPCPP_OPENCL_LIBRARY
)
