cmake_minimum_required(VERSION 3.21...3.27 FATAL_ERROR)

project(external_usm_allocation VERSION 0.1 LANGUAGES CXX
  DESCRIPTION "Example of passing external C++ USM allocation to Python")

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

set(py_module_name _external_usm_alloc)
set(_sources
  external_usm_allocation/_usm_alloc_example.cpp
)
pybind11_add_module(${py_module_name}
    MODULE
    ${_sources}
)
add_sycl_to_target(TARGET ${py_module_name} SOURCES ${_sources})
target_include_directories(${py_module_name} PUBLIC ${Dpctl_INCLUDE_DIRS})
install(TARGETS ${py_module_name}
  DESTINATION external_usm_allocation
)

set(ignoreMe "${SKBUILD}")
