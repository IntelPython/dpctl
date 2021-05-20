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
# The module defines a helper function that uses git to figure our project
# version. Note that for the version retrieval to work the tags must be defined
# using correct semantic versioning format.
#
# Example usage:
#
# get_version()
#
# If successful, the following variables will be defined in the parent scope:
#
# VERSION_MAJOR
# VERSION_MINOR
# VERSION_MINOR
# VERSION
# SEMVER
cmake_minimum_required( VERSION 3.14.0 )

function(get_version)
    # Use git describe to get latest tag name
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0
        RESULT_VARIABLE result
        OUTPUT_VARIABLE latest_tag
        ERROR_VARIABLE error
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )

    if(NOT result EQUAL 0)
        message(WARNING
            "Something went wrong when executing \"git describe\". "
            "Seting all version values to 0."
        )
        set(VERSION_MAJOR 0     PARENT_SCOPE)
        set(VERSION_MINOR 0     PARENT_SCOPE)
        set(VERSION_MINOR 0     PARENT_SCOPE)
        set(VERSION       0.0.0 PARENT_SCOPE)
        set(SEMVER        0.0.0 PARENT_SCOPE)
    endif()

    # Check if the tag naming follows semantic versioning scheme.
    if(latest_tag MATCHES "[0-9]+\.[0-9]+\.[0-9]+")
        string(REPLACE "." ";" VERSION_LIST1 "${latest_tag}")
        list(GET VERSION_LIST1 0 major)
        list(GET VERSION_LIST1 1 minor)
        list(GET VERSION_LIST1 2 patch)
    else()
        message(WARNING
            "The last git tag does not use proper semantic versioning. "
            "Seting all version values to 0."
        )
        set(VERSION_MAJOR 0     PARENT_SCOPE)
        set(VERSION_MINOR 0     PARENT_SCOPE)
        set(VERSION_PATCH 0     PARENT_SCOPE)
        set(VERSION       0.0.0 PARENT_SCOPE)
        set(SEMVER        0.0.0 PARENT_SCOPE)
        return()
    endif()

    # Use git describe to get the hash off latest tag
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags
        RESULT_VARIABLE result
        OUTPUT_VARIABLE latest_commit
        ERROR_VARIABLE error
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )

    if(NOT result EQUAL 0)
        message(WARNING
            "Something went wrong when executing \"git describe\". "
            "Seting all version values to 0."
        )
        set(VERSION_MAJOR 0     PARENT_SCOPE)
        set(VERSION_MINOR 0     PARENT_SCOPE)
        set(VERSION_PATCH 0     PARENT_SCOPE)
        set(VERSION       0.0.0 PARENT_SCOPE)
        set(SEMVER        0.0.0 PARENT_SCOPE)
    endif()

    if("${latest_tag}" STREQUAL "${latest_commit}")
        # We are at a tag and version and semver are both the same
        set(VERSION_MAJOR ${major} PARENT_SCOPE)
        set(VERSION_MINOR ${minor} PARENT_SCOPE)
        set(VERSION_PATCH ${patch} PARENT_SCOPE)
        set(VERSION ${latest_tag} PARENT_SCOPE)
        set(SEMVER ${latest_tag} PARENT_SCOPE)
    else()
        set(VERSION_MAJOR ${major} PARENT_SCOPE)
        set(VERSION_MINOR ${minor} PARENT_SCOPE)
        set(VERSION_PATCH ${patch} PARENT_SCOPE)
        set(VERSION ${latest_tag} PARENT_SCOPE)
        set(SEMVER ${latest_commit} PARENT_SCOPE)
    endif ()

endfunction(get_version)
