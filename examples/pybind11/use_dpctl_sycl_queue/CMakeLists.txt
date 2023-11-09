cmake_minimum_required(VERSION 3.21...3.27 FATAL_ERROR)

project(use_queue_device VERSION 0.1 LANGUAGES CXX
  DESCRIPTION "Example of using dpctl.SyclQueue <-> sycl::queue type caster")

set(DPCTL_CMAKE_MODULES_PATH "${CMAKE_SOURCE_DIR}/../../../cmake")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${DPCTL_CMAKE_MODULES_PATH})
find_package(IntelSYCL REQUIRED PATHS ${DPCTL_CMAKE_MODULES_PATH} NO_DEFAULT_PATH)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  URL https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.tar.gz
  URL_HASH SHA256=d475978da0cdc2d43b73f30910786759d593a9d8ee05b1b6846d1eb16c6d2e0c
)
FetchContent_MakeAvailable(pybind11)

find_package(Python REQUIRED COMPONENTS Development.Module NumPy)
find_package(Dpctl REQUIRED)

set(py_module_name _use_queue_device)
set(_sources use_queue_device/_example.cpp)
pybind11_add_module(${py_module_name}
    MODULE
    ${_sources}
)
add_sycl_to_target(TARGET ${py_module_name} SOURCES ${_sources})
target_include_directories(${py_module_name} PUBLIC ${Dpctl_INCLUDE_DIRS})
install(TARGETS ${py_module_name}
  DESTINATION use_queue_device
)

set(ignoreMe "${SKBUILD}")
