cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)
project("DPPPY-Oneapi-Interface - A lightweigth SYCL wrapper for Python libraries")

set (dpcpp_cmd "dpcpp")
set (dpcpp_arg "--version")

execute_process(
    COMMAND ${dpcpp_cmd} ${dpcpp_arg}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    RESULT_VARIABLE dpcpp_result
    OUTPUT_VARIABLE dpcpp_ver)

if(${dpcpp_result} MATCHES "0")
    string(REPLACE "\n" ";" DPCPP_VERSION_LIST "${dpcpp_ver}")
    list(GET DPCPP_VERSION_LIST 0 dpcpp_ver_line)
    foreach(X ${DPCPP_VERSION_LIST})
        message(STATUS "dpcpp ver[${dpcpp_result}]: ${X}")
    endforeach()
else()
    message(FATAL_ERROR "oneAPI needed to build DPPY-Oneapi-Interface")
endif()

# TODO check if the oneapi root exists

if(WIN32)
  set(CMAKE_CXX_COMPILER "dpcpp-cl")
  set(CMAKE_C_COMPILER "clang-cl")
else()
  set(CMAKE_CXX_COMPILER "dpcpp")
  set(CMAKE_C_COMPILER "clang")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}                                          \
     -Wall -Wextra -Winit-self -Wuninitialized -Wmissing-declarations          \
     -fdiagnostics-color=auto -pedantic-errors                                 \
  ")
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -ggdb3 -DDEBUG ")
endif()

# We need OpenCL support
find_package(OpenCL REQUIRED)

message(STATUS "OCL_INCLUDE_DIR: ${OpenCL_INCLUDE_DIR}")
message(STATUS "OCL_LIBRARY: ${OpenCL_LIBRARY}")

# Todo : Add build rules for unit tests

# Todo : Add build rules for doxygen

add_library(
  DDPYOneapiInterface
  SHARED
#  source/opencllite.cpp
  source/dppy_oneapi_interface.cpp
)

target_include_directories(
  DDPYOneapiInterface
  PRIVATE
  ${CMAKE_SOURCE_DIR}/include/
)

target_include_directories(
  DDPYOneapiInterface
  PUBLIC
  ${OpenCL_INCLUDE_DIR}
)

#target_include_directories(DDPYOneapiInterface PUBLIC ${LIBUSM_INCLUDE_DIR})

if (UNIX)
  target_include_directories(DDPYOneapiInterface PUBLIC ${DPCPP_INCLUDE_DIR})
endif()

install(
  TARGETS
  DDPYOneapiInterface
  LIBRARY DESTINATION
  "${CMAKE_INSTALL_PREFIX}/lib/"
)

# Install all headers
file(GLOB DPPY_ONEAPI_INTERFACE_HEADERS "${CMAKE_SOURCE_DIR}/include/*.h")
foreach(HEADER ${DPPY_ONEAPI_INTERFACE_HEADERS})
  install(FILES "${HEADER}" DESTINATION include)
endforeach()
