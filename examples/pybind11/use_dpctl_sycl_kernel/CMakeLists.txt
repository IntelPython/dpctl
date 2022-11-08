cmake_minimum_required(VERSION 3.21)

project(use_queue_device LANGUAGES CXX)

set(DPCTL_CMAKE_MODULES_PATH "${CMAKE_SOURCE_DIR}/../../../cmake")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${DPCTL_CMAKE_MODULES_PATH})
find_package(IntelDPCPP REQUIRED PATHS ${DPCTL_CMAKE_MODULES_PATH} NO_DEFAULT_PATH)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Debug)

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  URL https://github.com/pybind/pybind11/archive/refs/tags/v2.10.0.tar.gz
  URL_HASH SHA256=eacf582fa8f696227988d08cfc46121770823839fe9e301a20fbce67e7cd70ec
)
FetchContent_MakeAvailable(pybind11)

find_package(PythonExtensions REQUIRED)
find_package(Dpctl REQUIRED)
find_package(NumPy REQUIRED)

set(py_module_name _use_kernel)
pybind11_add_module(${py_module_name}
    MODULE
    use_kernel/_example.cpp
)
target_include_directories(${py_module_name} PUBLIC ${Dpctl_INCLUDE_DIRS})
install(TARGETS ${py_module_name}
  DESTINATION use_kernel
)

set(ignoreMe "${SKBUILD}")
