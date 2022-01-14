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
# The module uses git to clone the Level Zero source repository into the
# CMAKE_CURRENT_BINARY_DIR. The path to the Level Zero headers is then returned
# to the caller in the LEVEL_ZERO_INCLUDE_DIR variable.
#
# Example usage:
#
# get_level_zero_headers()
#
# If successful, the following variables will be defined in the parent scope:
#
# LEVEL_ZERO_INCLUDE_DIR

function(get_level_zero_headers)

    if(EXISTS level-zero)
      # Update the checkout
        execute_process(
            COMMAND ${GIT_EXECUTABLE} fetch
            RESULT_VARIABLE result
            ERROR_VARIABLE error
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/level-zero
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE
        )

        if(NOT result EQUAL 0)
            message(FATAL_ERROR
                "Could not update Level Zero sources."
            )
        endif()
    else()
        # Clone the Level Zero git repo
        execute_process(
            COMMAND ${GIT_EXECUTABLE} clone https://github.com/oneapi-src/level-zero.git
            RESULT_VARIABLE result
            ERROR_VARIABLE error
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE
        )

        if(NOT result EQUAL 0)
            message(FATAL_ERROR
                "Could not clone Level Zero sources from github.com/oneapi-src/level-zero."
            )
        endif()
    endif()

    # Use git describe to get latest tag name
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0
        RESULT_VARIABLE result
        OUTPUT_VARIABLE latest_tag
        ERROR_VARIABLE error
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/level-zero
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )

    if(NOT result EQUAL 0)
        message(FATAL_ERROR
            "Could not get the name for the latest release."
        )
    endif()

    # Use git describe to get latest tag name
    execute_process(
        COMMAND ${GIT_EXECUTABLE} checkout ${latest_tag}
        RESULT_VARIABLE result
        ERROR_VARIABLE error
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/level-zero
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )

    if(NOT result EQUAL 0)
        message(FATAL_ERROR
            "Could not checkout the latest release."
        )
    endif()

    # Populate the path to the headers
    find_path(LEVEL_ZERO_INCLUDE_DIR
        NAMES zet_api.h
        PATHS ${CMAKE_BINARY_DIR}/level-zero/include
        NO_DEFAULT_PATH
        NO_CMAKE_ENVIRONMENT_PATH
        NO_CMAKE_PATH
    )

    if(LEVEL_ZERO_INCLUDE_DIR STREQUAL "LEVEL_ZERO_INCLUDE_DIR-NOTFOUND")
        message(FATAL_ERROR
            "Could not find zet_api.h in cloned Level Zero repo."
        )
    else()
        message(STATUS
            "Level zero headers downloaded to: ${LEVEL_ZERO_INCLUDE_DIR}"
        )
    endif()

endfunction(get_level_zero_headers)
