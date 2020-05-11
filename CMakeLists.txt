cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)
project("DPPPY-Oneapi-Interface - A lightweigth SYCL wrapper for Python libraries")

if(ONEAPI_ROOT)
  set(ONEAPI_HOME "${ONEAPI_ROOT}")
  message(STATUS "ONEAPI_ROOT (read from command line): ${ONEAPI_HOME}")
elseif(DEFINED ENV{ONEAPI_ROOT})
  set(ONEAPI_HOME "$ENV{ONEAPI_ROOT}")
  message(STATUS "ONEAPI_ROOT (read from envar): ${ONEAPI_HOME}")
else()
  message(FATAL_ERROR "oneAPI needed to build DPPY-Oneapi-Interface")
endif()

# TODO check if the oneapi root exists

if(WIN32)
  set(CMAKE_CXX_COMPILER "dpcpp-cl")
  set(CMAKE_C_COMPILER "clang-cl")
else()
  # Populate the DPCPP-specific variables.
  # TODO These should be done more cleanly inside a FindDPCPP.cmake file.

  set(DPCPP_HOME ${ONEAPI_HOME}/compiler/latest)
  message(STATUS "DPCPP HOME: ${DPCPP_HOME}")
  set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH} ${DPCPP_HOME}/linux/bin/intel64")
  set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH} ${DPCPP_HOME}/linux/bin")
  set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH} ${DPCPP_HOME}/ioc/bin")

  set(DPCPP_INCLUDE_DIR "${DPCPP_HOME}/linux/include")
  set(DPCPP_LIBRARY "${DPCPP_HOME}/linux/lib")

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
