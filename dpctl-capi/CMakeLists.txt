cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project("dpCtl C API - A C wrapper for a subset of SYCL")

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
find_package(DPCPP REQUIRED)

if(DPCTL_ENABLE_LO_PROGRAM_CREATION)
    set(DPCTL_ENABLE_LO_PROGRAM_CREATION 1)
    find_package(LevelZero REQUIRED)
endif()

# Add the profiling flags if code coverage is required
if(DPCTL_GENERATE_COVERAGE)
    # check if llvm-cov and lcov are available
    find_package(Lcov REQUIRED)
    find_package(LLVMCov REQUIRED)
    # Turn on DPCTL_BUILD_CAPI_TESTS
    set(DPCTL_BUILD_CAPI_TESTS "ON")
    # Add profiling flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
    if(DPCTL_COVERAGE_REPORT_OUTPUT_DIR)
        set(COVERAGE_OUTPUT_DIR ${DPCTL_COVERAGE_REPORT_OUTPUT_DIR})
        message(STATUS "Saving coverage report to ${COVERAGE_OUTPUT_DIR}")
    else()
        set(COVERAGE_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
        message(STATUS "Saving coverage report to ${COVERAGE_OUTPUT_DIR}")
    endif()
endif()

configure_file(${CMAKE_SOURCE_DIR}/include/Config/dpctl_config.h.in
               ${CMAKE_SOURCE_DIR}/include/Config/dpctl_config.h)

if(WIN32)
    set(CMAKE_CXX_COMPILER:PATH "${DPCPP_ROOT}/bin/dpcpp")
    set(CMAKE_C_COMPILER:PATH "${DPCPP_ROOT}/bin/clang-cl")
    set(CMAKE_LINKER:PATH "${DPCPP_ROOT}/bin/lld-link")
    message(STATUS "Resetting CXX compiler to: " ${CMAKE_CXX_COMPILER})
    message(STATUS "Resetting C compiler to: " ${CMAKE_C_COMPILER})
    message(STATUS "Resetting Linker to: " ${CMAKE_LINK})
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

add_library(
  DPCTLSyclInterface
  SHARED
  source/dpctl_sycl_context_interface.cpp
  source/dpctl_sycl_device_interface.cpp
  source/dpctl_sycl_event_interface.cpp
  source/dpctl_sycl_kernel_interface.cpp
  source/dpctl_sycl_platform_interface.cpp
  source/dpctl_sycl_program_interface.cpp
  source/dpctl_sycl_queue_interface.cpp
  source/dpctl_sycl_queue_manager.cpp
  source/dpctl_sycl_usm_interface.cpp
  source/dpctl_utils.cpp
  helper/source/dpctl_utils_helper.cpp
)

if(DPCTL_BUILD_WITH_LEVEL_ZERO AND UNIX)
    target_include_directories(
      DPCTLSyclInterface
      PRIVATE
      ${CMAKE_SOURCE_DIR}/include/
      ${CMAKE_SOURCE_DIR}/helper/include/
      ${LEVEL_ZERO_INCLUDE_DIR}
    )
elseif(UNIX)
    target_include_directories(
      DPCTLSyclInterface
      PRIVATE
      ${CMAKE_SOURCE_DIR}/include/
      ${CMAKE_SOURCE_DIR}/helper/include/
    )
endif()

if(WIN32)
    message(
        STATUS
        "SYCL_INCLUDE_DIR: "
        ${DPCPP_SYCL_INCLUDE_DIR}
    )
    target_include_directories(
        DPCTLSyclInterface
        PUBLIC
        ${DPCPP_SYCL_INCLUDE_DIR}
    )
    target_link_libraries(
        DPCTLSyclInterface
        PRIVATE ${DPCPP_SYCL_LIBRARY}
        PRIVATE ${DPCPP_OPENCL_LIBRARY}
    )
endif()

install(
  TARGETS
  DPCTLSyclInterface
  LIBRARY DESTINATION
  "${CMAKE_INSTALL_PREFIX}/lib/"
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

# Install all headers in helper/include
file(GLOB HEADERS "${CMAKE_SOURCE_DIR}/helper/include/*.h")
foreach(HEADER ${HEADERS})
  install(FILES "${HEADER}" DESTINATION helper/include)
endforeach()

# Install all headers in include/Config
file(GLOB HEADERS "${CMAKE_SOURCE_DIR}/include/Config/*.h")
foreach(HEADER ${HEADERS})
  install(FILES "${HEADER}" DESTINATION include/Config)
endforeach()

# Enable to build the dpCtl C API test cases
if(DPCTL_BUILD_CAPI_TESTS)
    add_subdirectory(tests)
endif()
