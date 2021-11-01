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
# CMake find_package() module for the IntelSycl compiler and development
# environment.
#
# Example usage:
#
# find_package(IntelSycl)
#
# If successful, the following variables will be defined:
# IntelSycl_FOUND
# IntelSycl_VERSION
# IntelSycl_INCLUDE_DIR
# IntelSycl_C_COMPILER
# IntelSycl_CXX_COMPILER
# IntelSycl_SYCL_INCLUDE_DIR
# IntelSycl_LIBRARY_DIR
# IntelSycl_SYCL_LIBRARY
# IntelSycl_OPENCL_LIBRARY

include(FindPackageHandleStandardArgs)

# Check if a specific DPC++ installation directory was provided then set
# IntelSycl_ROOT to that path.
if(DPCTL_DPCPP_HOME_DIR)
    set(IntelSycl_ROOT ${DPCTL_DPCPP_HOME_DIR})
    if(NOT DPCTL_DPCPP_FROM_ONEAPI)
        message(STATUS
        "Not using standard oneAPI installation, but IntelSycl at "
        ${IntelSycl_ROOT}
        )
    endif()
# If DPC++ installation was not specified, check for ONEAPI_ROOT
elseif(DEFINED ENV{ONEAPI_ROOT})
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(IntelSycl_ROOT $ENV{ONEAPI_ROOT}/compiler/latest/windows)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(IntelSycl_ROOT $ENV{ONEAPI_ROOT}/compiler/latest/linux)
    else()
        message(FATAL_ERROR "Unsupported system.")
    endif()
else()
    message(FATAL_ERROR
        "Could not locate a DPC++ installation. Either pass the path to a "
        "custom location using DPCTL_DPCPP_HOME_DIR or set the "
        " ONEAPI_ROOT environment variable."
    )
    return()
endif()

# We will extract the version information from the compiler
set(clangxx_cmd "${IntelSycl_ROOT}/bin/clang++")
set(clangxx_arg "--version")

# Check if dpcpp is available
execute_process(
    COMMAND ${clangxx_cmd} ${clangxx_arg}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    RESULT_VARIABLE clangxx_result
    OUTPUT_VARIABLE clangxx_ver
)

# If dpcpp is found then set the package variables
if(${clangxx_result} MATCHES "0")
    string(REPLACE "\n" ";" IntelSycl_VERSION_LIST "${clangxx_ver}")
    set(IDX 0)
    foreach(X ${IntelSycl_VERSION_LIST})
        message(STATUS "dpcpp ver[${IDX}]: ${X}")
        MATH(EXPR IDX "${IDX}+1")
    endforeach()
    list(GET IntelSycl_VERSION_LIST 0 VERSION_STRING)

    # Get the dpcpp version
    string(REGEX MATCH
        "[0-9]+\.[0-9]+\.[0-9]+"
        IntelSycl_VERSION
        ${VERSION_STRING}
    )

    # Split out the version into major, minor an patch
    string(REPLACE "." ";" IntelSycl_VERSION_LIST1 "${IntelSycl_VERSION}")
    list(GET IntelSycl_VERSION_LIST1 0 IntelSycl_VERSION_MAJOR)
    list(GET IntelSycl_VERSION_LIST1 1 IntelSycl_VERSION_MINOR)
    list(GET IntelSycl_VERSION_LIST1 2 IntelSycl_VERSION_PATCH)
    set(IntelSycl_INCLUDE_DIR ${IntelSycl_ROOT}/include)
    set(IntelSycl_SYCL_INCLUDE_DIR ${IntelSycl_ROOT}/include/sycl)
    set(IntelSycl_LIBRARY_DIR ${IntelSycl_ROOT}/lib)
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(IntelSycl_SYCL_LIBRARY ${IntelSycl_LIBRARY_DIR}/sycl.lib)
        set(IntelSycl_OPENCL_LIBRARY ${IntelSycl_LIBRARY_DIR}/OpenCL.lib)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(IntelSycl_SYCL_LIBRARY ${IntelSycl_LIBRARY_DIR}/libsycl.so)
        set(IntelSycl_OPENCL_LIBRARY ${IntelSycl_LIBRARY_DIR}/libOpenCL.so)
    endif()

endif()

# Check if a specific version of DPCPP is requested.
if(IntelSycl_FIND_VERSION AND (DEFINED IntelSycl_VERSION))
    string(COMPARE
        LESS_EQUAL
        ${IntelSycl_FIND_VERSION}
        ${IntelSycl_VERSION}
        VERSION_MATCH
    )
    if(VERSION_MATCH)
        set(IntelSycl_FOUND TRUE)
    endif()
else()
    set(IntelSycl_FOUND TRUE)
endif()

find_package_handle_standard_args(IntelSycl DEFAULT_MSG
    IntelSycl_ROOT
    IntelSycl_FOUND
    IntelSycl_VERSION
    IntelSycl_INCLUDE_DIR
    IntelSycl_SYCL_INCLUDE_DIR
    IntelSycl_LIBRARY_DIR
    IntelSycl_SYCL_LIBRARY
    IntelSycl_OPENCL_LIBRARY
)
