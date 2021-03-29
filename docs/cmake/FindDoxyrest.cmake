include(FindPackageHandleStandardArgs)

message(STATUS "Doxyrest_DIR ${Doxyrest_DIR}")

find_file(DOXYREST_CONFIG_FILE
    NAMES doxyrest_config.cmake
    HINTS ${Doxyrest_DIR}/lib/cmake/doxyrest
          $ENV{Doxyrest_DIR}/lib/cmake/doxyrest
)
find_file(DOXYREST_VERSION_FILE
    NAMES doxyrest_version.cmake
    HINTS ${Doxyrest_DIR}/lib/cmake/doxyrest
          $ENV{Doxyrest_DIR}/lib/cmake/doxyrest
)

if(DOXYREST_CONFIG_FILE)
    include(${DOXYREST_CONFIG_FILE})
endif()

if(DOXYREST_VERSION_FILE)
    include(${DOXYREST_VERSION_FILE})
endif()

# Handle standard arguments to find_package like REQUIRED and QUIET
find_package_handle_standard_args(Doxyrest DEFAULT_MSG
    DOXYREST_EXE
    DOXYREST_FRAME_DIR
    DOXYREST_SPHINX_DIR
    DOXYREST_EXE
    DOXYREST_CMAKE_DIR
    DOXYREST_VERSION_MAJOR
    DOXYREST_VERSION_MINOR
    DOXYREST_VERSION_REVISION
    DOXYREST_VERSION_FULL
)
