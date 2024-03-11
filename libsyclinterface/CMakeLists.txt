cmake_minimum_required(VERSION 3.21...3.22 FATAL_ERROR)

project(
    "libDPCTLSYCLInterface"
    LANGUAGES C CXX
    DESCRIPTION "A C API for a subset of SYCL"
)

# Load our CMake modules to search for DPCPP and Level Zero
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/")

find_package(Git REQUIRED)

if(NOT DEFINED IntelSYCL_FOUND OR NOT IntelSYCL_FOUND)
   find_package(IntelSYCL REQUIRED)
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
if(DPCTL_DPCPP_FROM_ONEAPI)
    find_package(IntelSyclCompiler 2021.3.0 REQUIRED)
else()
    find_package(IntelSyclCompiler REQUIRED)
endif()

if(DPCTL_ENABLE_L0_PROGRAM_CREATION)
    set(DPCTL_ENABLE_L0_PROGRAM_CREATION 1)
    include(GetLevelZeroHeaders)
    get_level_zero_headers()
    if (UNIX)
        find_library(PI_LEVEL_ZERO_LIB
            NAMES pi_level_zero
            HINTS ${IntelSyclCompiler_LIBRARY_DIR}
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
        set(LIBZE_LOADER_FILENAME ${LIBZE_LOADER_FILENAME} PARENT_SCOPE)
    endif()
    set(LIBZE_LOADER_FILENAME ${LIBZE_LOADER_FILENAME} PARENT_SCOPE)
endif()

if (UNIX)
  find_library(PI_OPENCL_LIB
      NAMES pi_opencl
      HINTS ${IntelSyclCompiler_LIBRARY_DIR}
  )
  find_program(READELF_PROG readelf)
  find_program(GREP_PROG grep)
  execute_process(
    COMMAND ${READELF_PROG} -d ${PI_OPENCL_LIB}
    COMMAND ${GREP_PROG} OpenCL
    COMMAND ${GREP_PROG} -Po "libOpenCL[^\]]*"
    OUTPUT_VARIABLE LIBCL_LOADER_FILENAME
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
  )
  set(LIBCL_LOADER_FILENAME ${LIBCL_LOADER_FILENAME} PARENT_SCOPE)
endif()

message(STATUS "LIB_ZE: ${LIBZE_LOADER_FILENAME}")
message(STATUS "LIB_CL: ${LIBCL_LOADER_FILENAME}")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/syclinterface/Config/dpctl_config.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/include/syclinterface/Config/dpctl_config.h
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
        "-Wstrict-prototypes "
        "-Wno-unused-parameter "
    )
    string(CONCAT SDL_FLAGS
        "/GS "
	"/DynamicBase "
    )
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${WARNING_FLAGS} ${SDL_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNING_FLAGS} ${SDL_FLAGS}")
    set(CMAKE_C_FLAGS_DEBUG
        "${CMAKE_C_FLAGS_DEBUG} ${WARNING_FLAGS} -O0 -ggdb3 -DDEBUG"
    )
    set(CMAKE_CXX_FLAGS_DEBUG
        "${CMAKE_CXX_FLAGS_DEBUG} ${WARNING_FLAGS} -O0 -ggdb3 -DDEBUG"
    )
elseif(UNIX)
    string(CONCAT WARNING_FLAGS
        "-Wall "
        "-Wextra "
        "-Winit-self "
        "-Wunused-function "
        "-Wuninitialized "
        "-Wmissing-declarations "
        "-Wstrict-prototypes "
        "-Wno-unused-parameter "
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
#       "-fno-strict-overflow "  # implied by -fwrapv
        "-fno-delete-null-pointer-checks "
        "-fwrapv "
    )
    string(CONCAT CFLAGS
        "${WARNING_FLAGS}"
        "${SDL_FLAGS}"
    )
    string(CONCAT CXXFLAGS
        "${WARNING_FLAGS}"
        "${SDL_FLAGS}"
    )
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CFLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXXFLAGS}")
    set(CMAKE_C_FLAGS_DEBUG
        "${CMAKE_C_FLAGS_DEBUG} ${CFLAGS} -O0 -ggdb3 -DDEBUG"
    )
    set(CMAKE_CXX_FLAGS_DEBUG
        "${CMAKE_CXX_FLAGS_DEBUG} ${CXXFLAGS} -O0 -ggdb3 -DDEBUG"
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
add_sycl_to_target(TARGET DPCTLSyclInterface SOURCES ${sources} ${helper_sources})
# make fat binary
if(_dpctl_sycl_targets)
    target_compile_options(
        DPCTLSyclInterface
        PRIVATE
        -fsycl-targets=${_dpctl_sycl_targets}
    )
    target_link_options(
        DPCTLSyclInterface
        PRIVATE
        -fsycl-targets=${_dpctl_sycl_targets}
   )
endif()

if(DPCTL_GENERATE_COVERAGE)
    target_link_options(DPCTLSyclInterface
        PRIVATE -fprofile-instr-generate -fcoverage-mapping
    )
endif()

target_include_directories(DPCTLSyclInterface
    # PUBLIC
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/syclinterface
    ${CMAKE_CURRENT_SOURCE_DIR}/include/syclinterface/Support
    ${CMAKE_CURRENT_SOURCE_DIR}/include/syclinterface/Config
    # PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/include/
    ${SYCL_INCLUDE_DIR}
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

target_include_directories(DPCTLSyclInterface INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

install(TARGETS
    DPCTLSyclInterface
    LIBRARY
    DESTINATION ${CMAKE_INSTALL_PREFIX}/dpctl
)
install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/dpctl/include
    FILES_MATCHING REGEX "\\.h(pp)?$"
)

# Add sub-directory to build the dpctl C API test cases
if(DPCTL_BUILD_CAPI_TESTS)
    add_subdirectory(tests)
endif()
