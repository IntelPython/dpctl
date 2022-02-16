cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(example_use_mkl_gemm LANGUAGES CXX)
set(DPCTL_CMAKE_MODULES_PATH "${CMAKE_SOURCE_DIR}/../../../cmake")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${DPCTL_CMAKE_MODULES_PATH})
find_package(IntelDPCPP REQUIRED PATHS ${DPCTL_CMAKE_MODULES_PATH} NO_DEFAULT_PATH)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define CMAKE_INSTALL_xxx: LIBDIR, INCLUDEDIR
include(GNUInstallDirs)

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  URL https://github.com/pybind/pybind11/archive/refs/tags/v2.9.0.tar.gz
  URL_HASH SHA256=057fb68dafd972bc13afb855f3b0d8cf0fa1a78ef053e815d9af79be7ff567cb
)
FetchContent_MakeAvailable(pybind11)

find_package(PythonExtensions REQUIRED)
find_package(Dpctl REQUIRED)

find_library(mkl_core NAMES mkl_core PATH ${MKL_LIBRARY_DIR})
find_library(mkl_sycl NAMES mkl_sycl PATH ${MKL_LIBRARY_DIR})
find_library(mkl_intel_ilp64 NAMES mkl_intel_ilp64 PATH ${MKL_LIBRARY_DIR})
find_library(mkl_tbb_thread NAMES mkl_tbb_thread PATH ${MKL_LIBRARY_DIR})
find_library(tbb NAMES tbb PATH ${TBB_LIBRARY_DIR})

set(py_module_name _onemkl)

pybind11_add_module(${py_module_name}
    MODULE
    sycl_gemm/_onemkl.cpp
)
target_include_directories(${py_module_name}
    PUBLIC ${MKL_INCLUDE_DIR} ${TBB_INCLUDE_DIR}
)
target_link_libraries(${py_module_name}
    PUBLIC mkl_sycl mkl_intel_ilp64 mkl_tbb_thread mkl_core tbb
)

install(TARGETS ${py_module_name} DESTINATION sycl_gemm)
target_include_directories(${py_module_name} PUBLIC ${Dpctl_INCLUDE_DIRS})
target_link_directories(${py_module_name} PUBLIC ${MKL_LIBRARY_DIR} ${TBB_LIBRARY_DIR})

get_target_property(_sycl_gemm_sources ${py_module_name} SOURCES)
set_source_files_properties(${_sycl_gemm_sources}
  PROPERTIES
  COMPILE_OPTIONS "-O3;-Wno-deprecated-declarations"
)

set(ignoreMe "${SKBUILD}")
