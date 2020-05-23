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


if(WIN32)
  set(ONEAPI_HOME "$ENV{ONEAPI_ROOT}")
  message(STATUS "ONEAPI_ROOT (read from environment): ${ONEAPI_HOME}")
  set(CMAKE_CXX_COMPILER "dpcpp-cl")
  set(CMAKE_C_COMPILER "clang-cl")
  set(CMAKE_LINKER "lld-link")
  message(STATUS "Resetting CXX compiler to: " ${CMAKE_CXX_COMPILER})
  message(STATUS "Resetting C compiler to: " ${CMAKE_C_COMPILER})
  message(STATUS "Resetting Linker to: " ${CMAKE_LINK})
else()
  set(CMAKE_CXX_COMPILER "dpcpp")
  set(CMAKE_C_COMPILER "clang")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}                                          \
     -Wall -Wextra -Winit-self -Wuninitialized -Wmissing-declarations          \
     -fdiagnostics-color=auto                                                  \
  ")
endif()

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -ggdb3 -DDEBUG ")

# We need OpenCL support
find_package(OpenCL REQUIRED)

message(STATUS "OCL_INCLUDE_DIR : ${OpenCL_INCLUDE_DIR}")
message(STATUS "OCL_LIBRARY : ${OpenCL_LIBRARY}")

# Todo : Add build rules for unit tests

# Todo : Add build rules for doxygen

add_library(
  DPPYOneapiInterface
  SHARED
  source/dppy_oneapi_interface.cpp
)

# Install OpenCLlite as a separate library to be removed later
add_library(
  dpglue
  SHARED
  source/opencllite.c
)

target_include_directories(
  DPPYOneapiInterface
  PRIVATE
  ${CMAKE_SOURCE_DIR}/include/
)

target_include_directories(dpglue PRIVATE ${CMAKE_SOURCE_DIR}/include/)
target_include_directories(dpglue PUBLIC ${OpenCL_INCLUDE_DIR})

if (UNIX)
  target_include_directories(DPPYOneapiInterface PUBLIC ${DPCPP_INCLUDE_DIR})
elseif(WIN32)
  target_include_directories(DPPYOneapiInterface PUBLIC ${ONEAPI_HOME}/compiler/latest/windows/include/sycl)
  #target_link_libraries(DPPYOneapiInterface sycl)
endif()

install(
  TARGETS
  DPPYOneapiInterface
  LIBRARY DESTINATION
  "${CMAKE_INSTALL_PREFIX}/lib/"
)

install(
  TARGETS
  dpglue
  LIBRARY
  DESTINATION
  "${CMAKE_INSTALL_PREFIX}/lib/"
)

# Install all headers
file(GLOB HEADERS "${CMAKE_SOURCE_DIR}/include/*.h*")
foreach(HEADER ${HEADERS})
  install(FILES "${HEADER}" DESTINATION include)
endforeach()
