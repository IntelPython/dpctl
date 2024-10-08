cmake_minimum_required(VERSION 3.22...3.27 FATAL_ERROR)

project(example_use_mkl_gemm VERSION 0.1 LANGUAGES CXX
  DESCRIPTION "Example of using Python wrapper to oneMKL function")
set(DPCTL_CMAKE_MODULES_PATH "${CMAKE_SOURCE_DIR}/../../../cmake")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${DPCTL_CMAKE_MODULES_PATH})
find_package(IntelSYCL REQUIRED PATHS ${DPCTL_CMAKE_MODULES_PATH} NO_DEFAULT_PATH)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define CMAKE_INSTALL_xxx: LIBDIR, INCLUDEDIR
include(GNUInstallDirs)

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz
  URL_HASH SHA256=e08cb87f4773da97fa7b5f035de8763abc656d87d5773e62f6da0587d1f0ec20
  FIND_PACKAGE_ARGS NAMES pybind11
)
FetchContent_MakeAvailable(pybind11)

find_package(Python REQUIRED COMPONENTS Development.Module NumPy)
find_package(Dpctl REQUIRED)

find_library(mkl_core NAMES mkl_core PATHS ${MKL_LIBRARY_DIR} REQUIRED)
find_library(mkl_sycl NAMES mkl_sycl PATHS ${MKL_LIBRARY_DIR} REQUIRED)
find_library(mkl_intel_ilp64 NAMES mkl_intel_ilp64 PATHS ${MKL_LIBRARY_DIR} REQUIRED)
find_library(mkl_tbb_thread NAMES mkl_tbb_thread PATHS ${MKL_LIBRARY_DIR} REQUIRED)
find_library(tbb NAMES tbb PATHS ${TBB_LIBRARY_DIR} REQUIRED)
find_library(OpenCL NAMES OpenCL REQUIRED)

set(py_module_name _onemkl)

set(_sources sycl_gemm/_onemkl.cpp)
pybind11_add_module(${py_module_name}
    MODULE
    ${_sources}
)
add_sycl_to_target(TARGET ${py_module_name} SOURCES ${_sources})
target_compile_definitions(${py_module_name} PRIVATE -DMKL_ILP64)
target_include_directories(${py_module_name}
    PUBLIC ${MKL_INCLUDE_DIR} sycl_gemm
)
target_link_libraries(${py_module_name}
    PRIVATE ${mkl_sycl} ${mkl_intel_ilp64} ${mkl_tbb_thread} ${mkl_core} ${tbb}
)

install(TARGETS ${py_module_name} DESTINATION sycl_gemm)
target_include_directories(${py_module_name} PUBLIC ${Dpctl_INCLUDE_DIRS})

foreach(_src_fn ${_sources})
    get_source_file_property(_compile_options ${_src_fn} COMPILE_OPTIONS)
    set(_combined_options ${_compile_options} "-O3")
    set_source_files_properties(${_src_fn}
        PROPERTIES
        COMPILE_OPTIONS "${_combined_options}"
    )
endforeach()
target_link_options(${py_module_name} PRIVATE -fsycl-device-code-split=per_kernel)

add_executable(standalone_cpp
  EXCLUDE_FROM_ALL
  cpp/main.cpp
)
target_compile_options(standalone_cpp
  PRIVATE -O3 -Wno-deprecated-declarations
)
target_include_directories(standalone_cpp
  PUBLIC ${MKL_INCLUDE_DIR} ${TBB_INCLUDE_DIR} sycl_gemm
  )
target_link_libraries(standalone_cpp
    PRIVATE ${mkl_sycl} ${mkl_intel_ilp64} ${mkl_tbb_thread} ${mkl_core} ${tbb} ${OpenCL}
)

set(ignoreMe "${SKBUILD}")
