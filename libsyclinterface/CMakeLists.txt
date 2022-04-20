cmake_minimum_required(VERSION 3.10...3.22 FATAL_ERROR)

project(
    "libDPCTLSYCLInterface"
    DESCRIPTION "A C API for a subset of SYCL"
)

# Load our CMake modules to search for DPCPP and Level Zero
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/")

find_package(Git REQUIRED)

if(NOT DEFINED IntelDPCPP_FOUND OR NOT IntelDPCPP_FOUND)
   find_package(IntelDPCPP REQUIRED)
endif()

# Option to turn on support for creating Level Zero interoperability programs
# from a SPIR-V binary file.
option(DPCTL_ENABLE_L0_PROGRAM_CREATION
    "Enable Level Zero Program creation from SPIR-V"
    ON
)
# Option to generate code coverage report using llvm-cov and lcov.
option(DPCTL_GENERATE_COVERAGE
    "Build dpctl C API with coverage instrumentation"
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
# Option to turn on logging support for dpctl C API
option(DPCTL_ENABLE_GLOG
    "Enable the Google logging module"
    OFF
)

# Minimum version requirement only when oneAPI dpcpp is used.
find_package(IntelDPCPP REQUIRED)
if(DPCTL_DPCPP_FROM_ONEAPI)
    find_package(IntelSycl 2021.3.0 REQUIRED)
else()
    find_package(IntelSycl REQUIRED)
endif()

if(DPCTL_ENABLE_L0_PROGRAM_CREATION)
    set(DPCTL_ENABLE_L0_PROGRAM_CREATION 1)
    include(GetLevelZeroHeaders)
    get_level_zero_headers()
    if (UNIX)
        find_library(PI_LEVEL_ZERO_LIB
            NAMES pi_level_zero
            HINTS ${IntelSycl_LIBRARY_DIR}
        )
        find_program(READELF_PROG readelf)
        find_program(GREP_PROG grep)
        execute_process(
            COMMAND ${READELF_PROG} -d ${PI_LEVEL_ZERO_LIB}
            COMMAND ${GREP_PROG} libze_loader
            COMMAND ${GREP_PROG} -Po "libze_loader[^\]]*"
            OUTPUT_VARIABLE LIBZE_LOADER_FILENAME
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE
            )
    endif()
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Config/dpctl_config.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Config/dpctl_config.h
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
        "-Wno-deprecated-declarations "
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
        "-Wno-deprecated-declarations "
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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXXFLAGS}")
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

# Enable code coverage related settings
if(DPCTL_GENERATE_COVERAGE)
    include(SetupCoverage)
    setup_coverage_generation()

    # Turn on DPCTL_BUILD_CAPI_TESTS as building tests is needed to generate
    # coverage reports.
    set(DPCTL_BUILD_CAPI_TESTS "ON")
    if(DPCTL_COVERAGE_REPORT_OUTPUT_DIR)
        set(COVERAGE_OUTPUT_DIR ${DPCTL_COVERAGE_REPORT_OUTPUT_DIR})
    else()
        set(COVERAGE_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
    endif()
    message(STATUS "Coverage reports to be saved at ${COVERAGE_OUTPUT_DIR}")
endif()

add_library(DPCTLSyclInterface
    SHARED
    ${sources}
    ${helper_sources}
)

if(DPCTL_GENERATE_COVERAGE)
    target_link_options(DPCTLSyclInterface
        PRIVATE -fprofile-instr-generate -fcoverage-mapping
    )
endif()

target_include_directories(DPCTLSyclInterface
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Support
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Config
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/include/
    ${SYCL_INCLUDE_DIR}
)
target_link_libraries(DPCTLSyclInterface
  PRIVATE ${IntelSycl_SYCL_LIBRARY}
  PRIVATE ${IntelSycl_OPENCL_LIBRARY}
)

if(DPCTL_ENABLE_GLOG)
    find_package(glog REQUIRED)

    target_include_directories(DPCTLSyclInterface
        PRIVATE
        glog::glog
    )
    target_compile_definitions(DPCTLSyclInterface PRIVATE ENABLE_GLOG)
    target_link_libraries(DPCTLSyclInterface
        PRIVATE glog::glog
    )
endif()


include(GetProjectVersion)
# the get_version function is defined in the GetProjectVersion module and
# defines: VERSION, SEMVER, MAJOR, MINOR, PATCH. These variables are populated
# by parsing the output of git describe.
get_version()
set_target_properties(DPCTLSyclInterface
    PROPERTIES
        VERSION ${VERSION_MAJOR}.${VERSION_MINOR}
        SOVERSION ${VERSION_MAJOR}
)

if(DPCTL_ENABLE_L0_PROGRAM_CREATION)
    target_include_directories(DPCTLSyclInterface
        PRIVATE
        ${LEVEL_ZERO_INCLUDE_DIR}
    )
endif()

# Install all headers

file(GLOB MAIN_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
file(GLOB SUPPORT_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/Support/*.h")
file(GLOB CONFIG_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/Config/*.h")

set_target_properties(DPCTLSyclInterface
    PROPERTIES PUBLIC_HEADER
    "${MAIN_HEADERS}"
)

if (SKBUILD)
    set(_lib_destination dpctl)
    set(_include_destination dpctl/include/syclinterface)
else()
    set(_lib_destination ${CMAKE_INSTALL_PREFIX}/lib)
    set(_include_destination ${CMAKE_INSTALL_PREFIX}/include)
endif()

install(TARGETS
    DPCTLSyclInterface
    LIBRARY
    DESTINATION ${_lib_destination}
    ARCHIVE
    DESTINATION ${_lib_destination}
    RUNTIME
    DESTINATION ${_lib_destination}
    PUBLIC_HEADER
    DESTINATION ${_include_destination}
)
install(
    FILES ${SUPPORT_HEADERS}
    DESTINATION ${_include_destination}/Support
)
install(
    FILES ${CONFIG_HEADERS}
    DESTINATION ${_include_destination}/Config
)

# Add sub-directory to build the dpctl C API test cases
if(DPCTL_BUILD_CAPI_TESTS)
    add_subdirectory(tests)
endif()
