cmake_minimum_required(VERSION 3.16)

project(
  "libtensor"
  DESCRIPTION "DPC++ library implementing array operations"
)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")
find_package(Git REQUIRED)
find_package(Python COMPONENTS Development REQUIRED)
find_package(IntelSycl REQUIRED)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)

set(sources
     ${CMAKE_CURRENT_SOURCE_DIR}/source/sycl_kernels.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/source/dynamic_type_casting.cpp
)

add_library(tensor SHARED ${sources})

target_include_directories(tensor
    PRIVATE
    ${CMAKE_SOURCE_DIR}/../../include/
    ${CMAKE_SOURCE_DIR}/include/
    ${Python_INCLUDE_DIRS}
    ${IntelSycl_SYCL_INCLUDE_DIR}
)
target_link_libraries(tensor
    PRIVATE ${IntelSycl_SYCL_LIBRARY}
    PRIVATE ${IntelSycl_OPENCL_LIBRARY}
    PRIVATE ${Python_LIBRARIES}
)
