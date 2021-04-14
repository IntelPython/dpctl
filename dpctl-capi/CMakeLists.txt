cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project("dpctl C API - A C wrapper for a subset of SYCL")

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

# Load our CMake modules to search for DPCPP and Level Zero
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")
find_package(DPCPP 2021.2.0 REQUIRED)

if(DPCTL_ENABLE_LO_PROGRAM_CREATION)
    set(DPCTL_ENABLE_LO_PROGRAM_CREATION 1)
    find_package(LevelZero REQUIRED)
endif()

configure_file(${CMAKE_SOURCE_DIR}/include/Config/dpctl_config.h.in
               ${CMAKE_SOURCE_DIR}/include/Config/dpctl_config.h)

if(WIN32)
    set(CMAKE_CXX_COMPILER:PATH "${DPCPP_ROOT}/bin/dpcpp")
    set(CMAKE_C_COMPILER:PATH "${DPCPP_ROOT}/bin/clang-cl")
    set(CMAKE_LINKER:PATH "${DPCPP_ROOT}/bin/lld-link")
    message(STATUS "Resetting CXX compiler to: " ${CMAKE_CXX_COMPILER})
    message(STATUS "Resetting C compiler to: " ${CMAKE_C_COMPILER})
    message(STATUS "Resetting Linker to: " ${CMAKE_LINKER})
    set(WARNING_FLAGS "-Wall -Wextra -Winit-self -Wunused-function -Wuninitialized -Wmissing-declarations")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${WARNING_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNING_FLAGS} -Qstd=c++17")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${WARNING_FLAGS} -ggdb3 -DDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${WARNING_FLAGS} -ggdb3 -DDEBUG -Qstd=c++17")
elseif(UNIX)
    set(CMAKE_CXX_COMPILER:PATH "${DPCPP_ROOT}/bin/dpcpp")
    set(CMAKE_C_COMPILER:PATH "${DPCPP_ROOT}/bin/clang")
    set(CMAKE_LINKER:PATH "${DPCPP_ROOT}/bin/lld")
    set(SDL_FLAGS "-fstack-protector -fstack-protector-all -fpic -fPIC -D_FORTIFY_SOURCE=2 -Wformat -Wformat-security -fno-strict-overflow -fno-delete-null-pointer-checks")
    set(WARNING_FLAGS "-Wall -Wextra -Winit-self -Wunused-function -Wuninitialized -Wmissing-declarations -fdiagnostics-color=auto")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${WARNING_FLAGS} ${SDL_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNING_FLAGS} ${SDL_FLAGS} -std=c++17 -fsycl")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${WARNING_FLAGS} -ggdb3 -DDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${WARNING_FLAGS} -ggdb3 -DDEBUG -std=c++17 -fsycl")
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
    ${DPCPP_SYCL_INCLUDE_DIR}
)

target_link_libraries(DPCTLSyclInterface
    PRIVATE ${DPCPP_SYCL_LIBRARY}
    PRIVATE ${DPCPP_OPENCL_LIBRARY}
)

if(DPCTL_ENABLE_LO_PROGRAM_CREATION)
    if(UNIX)
        target_include_directories(DPCTLSyclInterface
            PRIVATE
            ${LEVEL_ZERO_INCLUDE_DIR}
        )
    elseif(WIN32)
        target_include_directories(DPCTLSyclInterface
            PRIVATE
            ${LEVEL_ZERO_INCLUDE_DIR}
        )
    else()
        message(WARNING
            "DPCTL support Level Zero program creation not supported "
            "on this system."
        )
    endif()
endif()

install(
  TARGETS DPCTLSyclInterface
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
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
    # check if llvm-cov and lcov are available
    find_package(Lcov REQUIRED)
    # These flags are set inside FindDPCPP
    if(NOT (${LLVM_COV_FOUND} AND ${LLVM_PROFDATA_FOUND}))
        message(FATAL_ERROR
            "llvm-cov and llvm-profdata are needed to generate coverage."
        )
    endif()
    # Turn on DPCTL_BUILD_CAPI_TESTS as building tests is needed to generate
    # coverage reports
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
