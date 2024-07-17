cmake_minimum_required(VERSION 3.21...3.27 FATAL_ERROR)

project(dpctl
    VERSION 0.18
    LANGUAGES CXX
    DESCRIPTION "Python interface for XPU programming"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
# set_property(GLOBAL PROPERTY GLOBAL_DEPENDS_DEBUG_MODE 1)

if (UNIX)
    add_link_options("-fuse-ld=lld")
endif()

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

add_library(DpctlCAPI INTERFACE)
target_include_directories(DpctlCAPI INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/dpctl/apis/include)
target_link_libraries(DpctlCAPI INTERFACE DPCTLSyclInterfaceHeaders)

install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/dpctl/apis/include/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/dpctl/include
    FILES_MATCHING REGEX "\\.h(pp)?$"
)

# Define CMAKE_INSTALL_xxx: LIBDIR, INCLUDEDIR
include(GNUInstallDirs)

# Fetch pybind11
include(FetchContent)

FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v2.12.0.tar.gz
    URL_HASH SHA256=bf8f242abd1abcd375d516a7067490fb71abd79519a282d22b6e4d19282185a7
    FIND_PACKAGE_ARGS NAMES pybind11
)
FetchContent_MakeAvailable(pybind11)

add_subdirectory(dpctl)

file(GLOB _cmake_scripts ${CMAKE_SOURCE_DIR}/cmake/*.cmake)
install(FILES ${_cmake_scripts}
    DESTINATION dpctl/resources/cmake
)
install(FILES
  ${CMAKE_SOURCE_DIR}/cmake/dpctl-config.cmake
  DESTINATION lib/cmake/dpctl
)

if (DPCTL_GENERATE_DOCS)
    add_subdirectory(docs)
endif()
