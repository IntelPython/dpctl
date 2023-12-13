cmake_minimum_required(VERSION 3.21...3.27 FATAL_ERROR)

project(dpctl
    VERSION 0.15
    LANGUAGES CXX
    DESCRIPTION "Python interface for XPU programming"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
# set_property(GLOBAL PROPERTY GLOBAL_DEPENDS_DEBUG_MODE 1)

# Option to generate code coverage report using llvm-cov and lcov.
option(DPCTL_GENERATE_COVERAGE
    "Build dpctl with coverage instrumentation"
    OFF
)
option(DPCTL_GENERATE_COVERAGE_FOR_PYBIND11_EXTENSIONS
    "Build dpctl pybind11 offloading extensions with coverage instrumentation"
    OFF
)
option(DPCTL_TARGET_CUDA
    "Build DPCTL to target CUDA devices"
    OFF
)

find_package(IntelSYCL REQUIRED PATHS ${CMAKE_SOURCE_DIR}/cmake NO_DEFAULT_PATH)

set(_dpctl_sycl_targets)
if ("x${DPCTL_SYCL_TARGETS}" STREQUAL "x")
   if(DPCTL_TARGET_CUDA)
      set(_dpctl_sycl_targets "nvptx64-nvidia-cuda,spir64-unknown-unknown")
   else()
      if(DEFINED ENV{DPCTL_TARGET_CUDA})
          set(_dpctl_sycl_targets "nvptx64-nvidia-cuda,spir64-unknown-unknown")
      endif()
   endif()
else()
   set(_dpctl_sycl_targets ${DPCTL_SYCL_TARGETS})
endif()

if(_dpctl_sycl_targets)
   message(STATUS "Compiling for -fsycl-targets=${_dpctl_sycl_targets}")
endif()

add_subdirectory(libsyclinterface)

file(GLOB _dpctl_capi_headers dpctl/apis/include/*.h*)
install(FILES ${_dpctl_capi_headers}
    DESTINATION dpctl/include
)

# Define CMAKE_INSTALL_xxx: LIBDIR, INCLUDEDIR
include(GNUInstallDirs)

# Fetch pybind11
include(FetchContent)

FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.tar.gz
    URL_HASH SHA256=d475978da0cdc2d43b73f30910786759d593a9d8ee05b1b6846d1eb16c6d2e0c
)
FetchContent_MakeAvailable(pybind11)

add_subdirectory(dpctl)

file(GLOB _cmake_scripts ${CMAKE_SOURCE_DIR}/cmake/*.cmake)
install(FILES ${_cmake_scripts}
    DESTINATION dpctl/resources/cmake
)

if (DPCTL_GENERATE_DOCS)
    add_subdirectory(docs)
endif()
