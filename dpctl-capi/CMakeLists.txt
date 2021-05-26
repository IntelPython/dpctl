cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(
    "libDPCTLSYCLInterface"
    DESCRIPTION "A C API for a subset of SYCL"
)

# Load our CMake modules to search for DPCPP and Level Zero
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")
find_package(Git REQUIRED)

option(DPCTL_CUSTOM_DPCPP_INSTALL_DIR
    "Use a custom version of DPCPP installed at the provided location."
    OFF
)
# Option to turn on support for creating Level Zero interoperability programs
# from a SPIR-V binary file.
option(DPCTL_ENABLE_LO_PROGRAM_CREATION
    "Enable Level Zero Program creation from SPIR-V"
    OFF
)
# Option to generate code coverage report using llvm-cov and lcov.
option(DPCTL_GENERATE_COVERAGE
    "Build dpctl C API with coverage instrumentation instrumentation"
    OFF
)
# Option to output html coverage report at a specific location.
option(DPCTL_COVERAGE_REPORT_OUTPUT_DIR
    "Save the generated lcov html report to the specified location"
    OFF
)
# Option to build the Gtests for dpctl C API
option(DPCTL_BUILD_CAPI_TESTS
    "Build dpctl C API google tests"
    OFF
)

# Minimum version requirement only when oneAPI dpcpp is used.
if(DPCTL_CUSTOM_DPCPP_INSTALL_DIR)
    find_package(IntelSycl REQUIRED)
else()
    find_package(IntelSycl 2021.2.0 REQUIRED)
endif()

if(DPCTL_ENABLE_LO_PROGRAM_CREATION)
    set(DPCTL_ENABLE_LO_PROGRAM_CREATION 1)
    include(GetLevelZeroHeaders)
    get_level_zero_headers()
endif()

configure_file(
    ${CMAKE_SOURCE_DIR}/include/Config/dpctl_config.h.in
    ${CMAKE_SOURCE_DIR}/include/Config/dpctl_config.h
)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)

if(WIN32)
    string(CONCAT WARNING_FLAGS
        "-Wall "
        "-Wextra "
        "-Winit-self "
        "-Wunused-function "
        "-Wuninitialized "
        "-Wmissing-declarations "
    )
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${WARNING_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNING_FLAGS}")
    set(CMAKE_C_FLAGS_DEBUG
        "${CMAKE_C_FLAGS_DEBUG} ${WARNING_FLAGS} -ggdb3 -DDEBUG"
    )
    set(CMAKE_CXX_FLAGS_DEBUG
        "${CMAKE_CXX_FLAGS_DEBUG} ${WARNING_FLAGS} -ggdb3 -DDEBUG"
    )
elseif(UNIX)
    string(CONCAT WARNING_FLAGS
        "-Wall "
        "-Wextra "
        "-Winit-self "
        "-Wunused-function "
        "-Wuninitialized "
        "-Wmissing-declarations "
        "-fdiagnostics-color=auto "
    )
    string(CONCAT SDL_FLAGS
        "-fstack-protector "
        "-fstack-protector-all "
        "-fpic "
        "-fPIC "
        "-D_FORTIFY_SOURCE=2 "
        "-Wformat "
        "-Wformat-security "
        "-fno-strict-overflow "
        "-fno-delete-null-pointer-checks "
    )
    string(CONCAT CFLAGS
        "${WARNING_FLAGS}"
        "${SDL_FLAGS}"
    )
    string(CONCAT CXXFLAGS
        "${WARNING_FLAGS}"
        "${SDL_FLAGS}"
        "-fsycl "
    )
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CFLAGS}")
    set(CMAKE_CXX_FLAGS "${CXXFLAGS}")
    set(CMAKE_C_FLAGS_DEBUG
        "${CMAKE_C_FLAGS_DEBUG} ${CFLAGS} -ggdb3 -DDEBUG"
    )
    set(CMAKE_CXX_FLAGS_DEBUG
        "${CMAKE_CXX_FLAGS_DEBUG} ${CXXFLAGS} -ggdb3 -DDEBUG"
    )
else()
    message(FATAL_ERROR "Unsupported system.")
endif()

file(GLOB_RECURSE sources
    ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp
)

# Exclude from sources
list(REMOVE_ITEM
    sources
    "${CMAKE_CURRENT_SOURCE_DIR}/source/dpctl_vector_templ.cpp"
)

file(GLOB_RECURSE helper_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/source/*.cpp
)

add_library(DPCTLSyclInterface
    SHARED
    ${sources}
    ${helper_sources}
)

target_include_directories(DPCTLSyclInterface
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include/
    ${CMAKE_SOURCE_DIR}/helper/include/
    ${IntelSycl_SYCL_INCLUDE_DIR}
)

target_link_libraries(DPCTLSyclInterface
    PRIVATE ${IntelSycl_SYCL_LIBRARY}
    PRIVATE ${IntelSycl_OPENCL_LIBRARY}
)

if(DPCTL_ENABLE_LO_PROGRAM_CREATION)
    target_include_directories(DPCTLSyclInterface
        PRIVATE
        ${LEVEL_ZERO_INCLUDE_DIR}
    )
endif()

# FIXME: Turning off for release until conda build can be packaging symbolic links
# NOTE: Till we hit 1.0.0 we will keep using the MINOR version to set the API
# version of the library.
# include(GetProjectVersion)
# the get_version function is defined in the GetProjectVersion module and
# defines: VERSION, SEMVER, MAJOR, MINOR, PATCH. These variables are populated
# by parsing the output of git describe.
# get_version()
# set_target_properties(DPCTLSyclInterface PROPERTIES VERSION ${VERSION_MINOR})
# set_target_properties(DPCTLSyclInterface PROPERTIES SOVERSION 1)

install(TARGETS
    DPCTLSyclInterface
    LIBRARY
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)

# Install all headers
file(GLOB HEADERS "${CMAKE_SOURCE_DIR}/include/*.h")
foreach(HEADER ${HEADERS})
  install(FILES "${HEADER}" DESTINATION include)
endforeach()

# Install all headers in include/Support
file(GLOB HEADERS "${CMAKE_SOURCE_DIR}/include/Support/*.h")
foreach(HEADER ${HEADERS})
  install(FILES "${HEADER}" DESTINATION include/Support)
endforeach()

# Install all headers in include/Config
file(GLOB HEADERS "${CMAKE_SOURCE_DIR}/include/Config/*.h")
foreach(HEADER ${HEADERS})
  install(FILES "${HEADER}" DESTINATION include/Config)
endforeach()

# Enable code coverage related settings
if(DPCTL_GENERATE_COVERAGE)
    # check if lcov is available
    find_package(Lcov REQUIRED)
    # check if llvm-cov version 11 is available
    find_package(LLVMCov 11 REQUIRED)
    # check if llvm-profdata is available
    find_package(LLVMProfdata REQUIRED)
    # Turn on DPCTL_BUILD_CAPI_TESTS as building tests is needed to generate
    # coverage reports.
    set(DPCTL_BUILD_CAPI_TESTS "ON")
    if(DPCTL_COVERAGE_REPORT_OUTPUT_DIR)
        set(COVERAGE_OUTPUT_DIR ${DPCTL_COVERAGE_REPORT_OUTPUT_DIR})
        message(STATUS "Coverage reports to be saved at ${COVERAGE_OUTPUT_DIR}")
    else()
        set(COVERAGE_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
        message(STATUS "Coverage reports to be saved at ${COVERAGE_OUTPUT_DIR}")
    endif()
endif()

# Add sub-directory to build the dpctl C API test cases
if(DPCTL_BUILD_CAPI_TESTS)
    add_subdirectory(tests)
endif()
