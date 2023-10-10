cmake_minimum_required(VERSION 3.21...3.27 FATAL_ERROR)

project(use_queue_device VERSION 0.1 LANGUAGES CXX
  DESCRIPTION "Example of using dpctl.program.SyclKernel <-> sycl::kernel type casting")

set(DPCTL_CMAKE_MODULES_PATH "${CMAKE_SOURCE_DIR}/../../../cmake")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${DPCTL_CMAKE_MODULES_PATH})
find_package(IntelSYCL REQUIRED PATHS ${DPCTL_CMAKE_MODULES_PATH} NO_DEFAULT_PATH)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Debug)

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  URL https://github.com/pybind/pybind11/archive/refs/tags/v2.10.2.tar.gz
  URL_HASH SHA256=93bd1e625e43e03028a3ea7389bba5d3f9f2596abc074b068e70f4ef9b1314ae
)
FetchContent_MakeAvailable(pybind11)

find_package(Python REQUIRED COMPONENTS Development.Module NumPy)
find_package(Dpctl REQUIRED)

set(py_module_name _use_kernel)
set(_sources use_kernel/_example.cpp)
pybind11_add_module(${py_module_name}
    MODULE
    ${_sources}
)
add_sycl_to_target(TARGET ${py_module_name} SOURCES ${_sources})
target_include_directories(${py_module_name} PUBLIC ${Dpctl_INCLUDE_DIRS})
install(TARGETS ${py_module_name}
  DESTINATION use_kernel
)

set(ignoreMe "${SKBUILD}")
