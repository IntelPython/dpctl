cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project("Data-parallel Control (dpCtl)")
# Add the cmake folder so the FindSphinx module is found
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(Sphinx REQUIRED)
find_package(Doxygen REQUIRED)
find_package (Git)

set(DOC_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/generated_docs)
if( DPCTL_DOCGEN_PREFIX )
    message(STATUS "Generating dpCtl documents in " ${DPCTL_DOCGEN_PREFIX})
    set(DOC_OUTPUT_DIR ${DPCTL_DOCGEN_PREFIX})
endif()

set(CURRENT_RELEASE "")

# Use git describe to get latest tag name
if (GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0
        RESULT_VARIABLE result
        OUTPUT_VARIABLE CURRENT_RELEASE
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(CURRENT_COMMIT "")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags
        RESULT_VARIABLE result
        OUTPUT_VARIABLE CURRENT_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (NOT "${CURRENT_RELEASE}" STREQUAL "${CURRENT_COMMIT}")
        set(CURRENT_RELEASE "master")
    endif ()
endif (GIT_FOUND)

set(DOXYGEN_INPUT_DIR ../dpctl-capi)
set(DOXYGEN_OUTPUT_DIR ${DOC_OUTPUT_DIR}/doxygen)
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)
set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

# Replace variables inside @@ with the current values
configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})

add_custom_command(
    OUTPUT ${DOXYGEN_INDEX_FILE}
    DEPENDS ${DPCTL_PUBLIC_HEADERS}
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
    COMMENT "Generating Doxygen documentation"
    VERBATIM
)

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_OUTPUT_DIR ${DOC_OUTPUT_DIR}/docs)
set(SPHINX_INDEX_FILE ${SPHINX_OUTPUT_DIR}/index.html)
set(SPHINX_CONF_IN ${SPHINX_SOURCE}/conf.in)
set(SPHINX_CONF_OUT ${SPHINX_SOURCE}/conf.py)

# Create a conf.py by replacing variables inside @@ with the current values
configure_file(${SPHINX_CONF_IN} ${SPHINX_CONF_OUT} @ONLY)

# Only regenerate Sphinx when:
# - Doxygen has rerun
# - Our doc files have been updated
# - The Sphinx config has been updated
add_custom_command(
    OUTPUT ${SPHINX_INDEX_FILE}
    COMMAND
        ${SPHINX_EXECUTABLE} -b html
        # Tell Breathe where to find the Doxygen output
        -Dbreathe_projects.dpCtl-CAPI=${DOXYGEN_OUTPUT_DIR}/xml
        ${SPHINX_SOURCE} ${SPHINX_OUTPUT_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
    # Other docs files that can be edited manually
    ${CMAKE_CURRENT_SOURCE_DIR}/index.rst
    ${DOXYGEN_INDEX_FILE}
    MAIN_DEPENDENCY ${SPHINX_CONF_OUT} ${SPHINX_CONF_IN}
    COMMENT "Generating Sphinx documentation"
)

# Target to generate only Doxygen documentation
add_custom_target(
    Doxygen
    ALL
    DEPENDS ${DOXYGEN_INDEX_FILE}
)

# Target to generate all documentation Sphinx and Doxygen
add_custom_target(
    Sphinx
    ALL
    DEPENDS Doxygen ${SPHINX_INDEX_FILE}
)
