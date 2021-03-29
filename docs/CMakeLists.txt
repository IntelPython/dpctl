cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project("Data-parallel Control (dpctl)")

# Option to generate rst for C API and add to Sphinx documentation
option(DPCTL_ENABLE_DOXYREST
    "Enable generation of rst files for C API"
    OFF
)

# This function defines everything needed to generate Doxygen docs
function(_setup_doxygen)
    # We generate doxygen only for the public headers to keep the Doxyrest
    # generated rst files clean.
    # FIXME: make it possible to generate doxygen for all files.
    set(DOXYGEN_INPUT_DIR ../dpctl-capi/include)
    set(DOXYGEN_OUTPUT_DIR ${DOC_OUTPUT_DIR}/doxygen)
    set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/xml/index.xml)
    set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    # Populate the configurable values in the Doxyfile.in and
    # generate the actual Doxyfile.
    configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

    file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})

    # Custom command to run Doxygen
    add_custom_command(
        OUTPUT ${DOXYGEN_INDEX_FILE}
        DEPENDS ${DPCTL_PUBLIC_HEADERS}
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
        COMMENT "Generating Doxygen documentation"
        VERBATIM
    )

    # Target to generate only Doxygen documentation
    add_custom_target(
        Doxygen
        ALL
        DEPENDS ${DOXYGEN_INDEX_FILE}
    )
endfunction()

function(_setup_doxyrest)
    set(DOXYREST_OUTPUT_DIR_NAME dpctl-capi-doxyrest)
    set(DOXYREST_OUTPUT_DIR
        ${CMAKE_CURRENT_SOURCE_DIR}/${DOXYREST_OUTPUT_DIR_NAME}
    )
    set(DOXYREST_CONFIG_IN ${CMAKE_CURRENT_SOURCE_DIR}/doxyrest-config.lua.in)
    set(DOXYREST_CONFIG_OUT ${CMAKE_CURRENT_SOURCE_DIR}/doxyrest-config.lua)
    set(DOXYREST_OUTPUT ${DOXYREST_OUTPUT_DIR}/index.rst)
    set(DOXYGEN_OUTPUT_DIR ${DOC_OUTPUT_DIR}/doxygen)
    configure_file(${DOXYREST_CONFIG_IN} ${DOXYREST_CONFIG_OUT} @ONLY)
    configure_file(${INDEX_DOXYREST_IN} ${INDEX_OUT} @ONLY)
    add_custom_command(
        OUTPUT ${DOXYREST_OUTPUT}
        COMMAND
            ${DOXYREST_EXE} -c
            ${DOXYREST_CONFIG_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS
        # Other docs files that can be edited manually
        ${INDEX_OUT}
        ${DOXYGEN_INDEX_FILE}
        MAIN_DEPENDENCY ${DOXYREST_CONFIG_OUT} ${DOXYREST_CONFIG_IN}
        COMMENT "Generating Doxyrest documentation"
    )
    # Target to generate rst from Doxygen XML using Doxyrest
    add_custom_target(
        Doxyrest
        ALL
        DEPENDS Doxygen ${DOXYREST_OUTPUT}
    )
endfunction()

function(_setup_sphinx)
    set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
    set(SPHINX_OUTPUT_DIR ${DOC_OUTPUT_DIR}/docs)
    set(SPHINX_INDEX_FILE ${SPHINX_OUTPUT_DIR}/index.html)
    set(SPHINX_CONF_IN ${SPHINX_SOURCE}/conf.in)
    set(SPHINX_CONF_OUT ${SPHINX_SOURCE}/conf.py)

    # Only regenerate Sphinx when:
    # - Doxygen has rerun
    # - Our doc files have been updated
    # - The Sphinx config has been updated
    if(DPCTL_ENABLE_DOXYREST)
        add_custom_command(
            OUTPUT ${SPHINX_INDEX_FILE}
            COMMAND
                ${SPHINX_EXECUTABLE} -b html
                ${SPHINX_SOURCE}
                ${SPHINX_OUTPUT_DIR}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS
            # Other docs files that can be edited manually
            ${CMAKE_CURRENT_SOURCE_DIR}/index.rst
            ${DOXYGEN_INDEX_FILE}
            MAIN_DEPENDENCY ${SPHINX_CONF_OUT} ${SPHINX_CONF_IN}
            COMMENT "Generating Sphinx documentation"
        )
        # Target to generate Sphinx
        add_custom_target(
            Sphinx
            ALL
            DEPENDS Doxyrest ${SPHINX_INDEX_FILE}
        )
    else()
        configure_file(${INDEX_NO_DOXYREST_IN} ${INDEX_OUT} @ONLY)
        add_custom_command(
            OUTPUT ${SPHINX_INDEX_FILE}
            COMMAND
                ${SPHINX_EXECUTABLE} -b html
                ${SPHINX_SOURCE}
                ${SPHINX_OUTPUT_DIR}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS
            # Other docs files that can be edited manually
            ${CMAKE_CURRENT_SOURCE_DIR}/index.rst
            MAIN_DEPENDENCY ${SPHINX_CONF_OUT} ${SPHINX_CONF_IN}
            COMMENT "Generating Sphinx documentation"
        )
        # Target to generate Sphinx
        add_custom_target(
            Sphinx
            ALL
            DEPENDS ${SPHINX_INDEX_FILE}
        )
    endif()
    # Create a conf.py by replacing variables inside @@ with the current values
    configure_file(${SPHINX_CONF_IN} ${SPHINX_CONF_OUT} @ONLY)
endfunction()

function(_set_current_release)
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
endfunction()

##----------------------------------------------------------------------------##

# Add the cmake folder so the FindSphinx module is found
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
find_package(Git)
find_package(Sphinx REQUIRED)
find_package(Doxygen REQUIRED)
if (DPCTL_ENABLE_DOXYREST)
    find_package(Lua REQUIRED)
    find_package(Doxyrest REQUIRED)
endif()

# Set the location where the generated docs are saved
if(DPCTL_DOCGEN_PREFIX)
    message(STATUS "Generating dpctl documents in " ${DPCTL_DOCGEN_PREFIX})
    set(DOC_OUTPUT_DIR ${DPCTL_DOCGEN_PREFIX})
else()
    set(DOC_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/generated_docs)
endif()

set(INDEX_NO_DOXYREST_IN ${CMAKE_CURRENT_SOURCE_DIR}/index_no_doxyrest.rst.in)
set(INDEX_DOXYREST_IN ${CMAKE_CURRENT_SOURCE_DIR}/index_doxyrest.rst.in)
set(INDEX_OUT ${CMAKE_CURRENT_SOURCE_DIR}/index.rst)

_set_current_release()
_setup_doxygen()
if(DPCTL_ENABLE_DOXYREST)
    _setup_doxyrest()
endif()
_setup_sphinx()
