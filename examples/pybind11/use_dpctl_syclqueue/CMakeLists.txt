cmake_minimum_required(VERSION 3.21)

project(use_queue_device LANGUAGES CXX)

set(DPCTL_CMAKE_MODULES_PATH "${CMAKE_SOURCE_DIR}/../../../cmake")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${DPCTL_CMAKE_MODULES_PATH})
find_package(IntelDPCPP REQUIRED PATHS ${DPCTL_CMAKE_MODULES_PATH} NO_DEFAULT_PATH)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  URL https://github.com/pybind/pybind11/archive/refs/tags/v2.9.2.tar.gz
  URL_HASH SHA256=6bd528c4dbe2276635dc787b6b1f2e5316cf6b49ee3e150264e455a0d68d19c1
)
FetchContent_MakeAvailable(pybind11)

find_package(PythonExtensions REQUIRED)
find_package(Dpctl REQUIRED)
find_package(NumPy REQUIRED)

set(py_module_name _use_queue_device)
pybind11_add_module(${py_module_name}
    MODULE
    use_queue_device/_example.cpp
)
target_include_directories(${py_module_name} PUBLIC ${Dpctl_INCLUDE_DIRS})
install(TARGETS ${py_module_name}
  DESTINATION use_queue_device
)

set(ignoreMe "${SKBUILD}")
