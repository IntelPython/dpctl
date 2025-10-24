cmake_minimum_required(VERSION 3.21...3.27 FATAL_ERROR)

project(dpctl
    VERSION 0.22
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

set(DPCTL_TARGET_CUDA
    ""
    CACHE STRING
    "Build DPCTL to target CUDA device. \
Set to a truthy value (e.g., ON, TRUE) to use default architecture (sm_50), \
or to a specific architecture like sm_80."
)
set(DPCTL_TARGET_HIP
    ""
    CACHE STRING
    "Build DPCTL to target a HIP device architecture"
)
option(
     DPCTL_WITH_REDIST
     "Build DPCTL assuming DPC++ redistributable is installed into Python prefix"
     OFF)
option(
     DPCTL_OFFLOAD_COMPRESS
     "Build using offload section compression feature of DPC++ to reduce \
size of shared object with offloading sections"
     OFF
)

find_package(IntelSYCL REQUIRED PATHS ${CMAKE_SOURCE_DIR}/cmake NO_DEFAULT_PATH)

set(_dpctl_sycl_target_compile_options)
set(_dpctl_sycl_target_link_options)

set(_dpctl_sycl_targets)
set(_dpctl_cuda_arch)

if ("x${DPCTL_SYCL_TARGETS}" STREQUAL "x")
   if (DPCTL_TARGET_CUDA)
      if(DPCTL_TARGET_CUDA MATCHES "^sm_")
         set(_dpctl_cuda_arch ${DPCTL_TARGET_CUDA})
      elseif(DPCTL_TARGET_CUDA MATCHES "^(ON|TRUE|YES|Y|1)$")
         set(_dpctl_cuda_arch "sm_50")
      else()
         message(FATAL_ERROR
            "Invalid value for DPCTL_TARGET_CUDA: \"${DPCTL_TARGET_CUDA}\". "
            "Expected 'ON', 'TRUE', 'YES', 'Y', '1', or a CUDA architecture like 'sm_80'."
         )
      endif()
      set(_dpctl_sycl_targets "nvidia_gpu_${_dpctl_cuda_arch},spir64-unknown-unknown")
   endif()

   if (DPCTL_TARGET_HIP)
      if(DPCTL_TARGET_HIP MATCHES "^gfx")
         if(_dpctl_sycl_targets)
            set(_dpctl_sycl_targets "amd_gpu_${DPCTL_TARGET_HIP},${_dpctl_sycl_targets}")
         else()
            set(_dpctl_sycl_targets "amd_gpu_${DPCTL_TARGET_HIP},spir64-unknown-unknown")
         endif()
      else()
         message(FATAL_ERROR
            "Invalid value for DPCTL_TARGET_HIP: \"${DPCTL_TARGET_HIP}\". "
            "Expected something starting with 'gfx', e.g. 'gfx1030'."
         )
      endif()
   endif()
else()
   if ("${DPCTL_SYCL_TARGETS}" MATCHES "amdgcn-amd-amdhsa")
      message(FATAL_ERROR
         "Legacy target 'amdgcn-amd-amdhsa' is not supported. "
         "Use alias form 'amd_gpu_<arch>' instead."
      )
   endif()
   set(_dpctl_sycl_targets ${DPCTL_SYCL_TARGETS})
endif()

if (_dpctl_sycl_targets)
   message(STATUS "Compiling for -fsycl-targets=${_dpctl_sycl_targets}")
   list(APPEND _dpctl_sycl_target_compile_options -fsycl-targets=${_dpctl_sycl_targets})
   list(APPEND _dpctl_sycl_target_link_options -fsycl-targets=${_dpctl_sycl_targets})
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

# find Python before enabling pybind11
find_package(Python REQUIRED COMPONENTS Development.Module)

# Define CMAKE_INSTALL_xxx: LIBDIR, INCLUDEDIR
include(GNUInstallDirs)

# Fetch pybind11
include(FetchContent)

FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v3.0.1.tar.gz
    URL_HASH SHA256=741633da746b7c738bb71f1854f957b9da660bcd2dce68d71949037f0969d0ca
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
