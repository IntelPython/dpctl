cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)
project("DPPPY-Oneapi-Interface - A lightweight SYCL wrapper for Python")

set(ONEAPI_ROOT "$ENV{ONEAPI_ROOT}")
string(COMPARE EQUAL "${ONEAPI_ROOT}" "" ONEAPI_ROOT_EMPTY)
if(${ONEAPI_ROOT_EMPTY})
    message(FATAL_ERROR "Please set an environment varaible ONEAPI_ROOT \
                         pointing to you oneAPI install directory.")
else()
    message(STATUS "oneAPI Root:" ${ONEAPI_ROOT})
endif()

if(WIN32)
    set(DPCPP_ROOT "${ONEAPI_ROOT}/compiler/latest/windows")
elseif(UNIX)
    set(DPCPP_ROOT "${ONEAPI_ROOT}/compiler/latest/linux")
endif()

if(WIN32)
    set (dpcpp_cmd "${DPCPP_ROOT}/bin/dpcpp-cl")
    set (dpcpp_arg "--version")
elseif(UNIX)
    set (dpcpp_cmd "${DPCPP_ROOT}/bin/dpcpp")
    set (dpcpp_arg "--version")
else()
    message(FATAL_ERROR "Unsupported system.")
endif()

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
    message(FATAL_MESSAGE "DPCPP needed to build DPPY-Oneapi-Interface")
endif()

if(WIN32)
    set(CMAKE_CXX_COMPILER:PATH "${DPCPP_ROOT}/bin/dpcpp-cl")
    set(CMAKE_C_COMPILER:PATH "${DPCPP_ROOT}/bin/clang-cl")
    set(CMAKE_LINKER:PATH "${DPCPP_ROOT}/bin/lld-link")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}                                        \
       -Wall -Wextra -Winit-self -Wuninitialized -Wmissing-declarations        \
       -fdiagnostics-color=auto -O3                                            \
    ")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -ggdb3 -DDEBUG ")
    message(STATUS "Resetting CXX compiler to: " ${CMAKE_CXX_COMPILER})
    message(STATUS "Resetting C compiler to: " ${CMAKE_C_COMPILER})
    message(STATUS "Resetting Linker to: " ${CMAKE_LINK})
elseif(UNIX)
    set(CMAKE_CXX_COMPILER "${DPCPP_ROOT}/bin/dpcpp")
    set(CMAKE_C_COMPILER "${DPCPP_ROOT}/bin/clang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}                                        \
       -Wall -Wextra -Winit-self -Wuninitialized -Wmissing-declarations        \
       -fdiagnostics-color=auto -O3                                            \
    ")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -ggdb3 -DDEBUG ")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
else()
    message(FATAL_ERROR "Unsupported system.")
endif()


#set(OPENCL_LIBDIR "$ENV{OpenCL_LIBDIR}")
#string(COMPARE EQUAL "${OPENCL_LIBDIR}" "" OPENCL_LIBDIR_EMPTY)

# This is a temporary hack to pass in the OpenCL install directory
# using an environment variable. We are doing this as we want to pass
# OpenCL_LIBDR to build.sh.

# if(${OPENCL_LIBDIR_EMPTY})
#     message(FATAL_ERROR "Set the OpenCL_LIBDIR and OpenCL_INCLUDE_DIR  \
#                    environment variables to point to an OpenCL installation.")
# endif()

#---- We need OpenCL
find_package(OpenCL REQUIRED)
if(WIN32)
    set(OpenCL_LIBRARY ${OpencL_LIBDIR}/libOpenCL.lib)
elseif(UNIX)
    set(OpenCL_LIBRARY ${OpencL_LIBDIR}/libOpenCL.so)
endif()

message(STATUS "OpenCL_INCLUDE_DIR : ${OpenCL_INCLUDE_DIR}")
message(STATUS "OpenCL_LIBRARY : ${OpenCL_LIBRARY}")

# Todo : Add build rules for unit tests

# Todo : Add build rules for doxygen

add_library(
  DPPYOneapiInterface
  SHARED
  source/dppy_oneapi_interface.cpp
  source/dppy_device_array.cpp
)

# Install dpglue as a separate library to be removed later
add_library(
  dpglue
  STATIC
  source/opencllite.c
)

target_include_directories(
  DPPYOneapiInterface
  PRIVATE
  ${CMAKE_SOURCE_DIR}/include/
)

target_include_directories(dpglue PRIVATE ${CMAKE_SOURCE_DIR}/include/)
target_include_directories(dpglue PUBLIC ${OpenCL_INCLUDE_DIR})

if(WIN32)
    message(
        STATUS
        "SYCL_INCLUDE_DIR : "
        ${ONEAPI_ROOT}/compiler/latest/windows/include/sycl
    )
    target_include_directories(
        DPPYOneapiInterface
        PUBLIC
        ${ONEAPI_ROOT}/compiler/latest/windows/include/sycl
    )
    target_link_libraries(
        DPPYOneapiInterface
        PRIVATE
        ${ONEAPI_ROOT}/compiler/latest/windows/lib/sycl.lib
    )
    target_link_libraries(
        dpglue
        PRIVATE
        ${OpenCL_INCLUDE_DIR}/../lib/x64/OpenCL.lib
    )
elseif(UNIX)
    message(
        STATUS
        "SYCL_INCLUDE_DIR : "
        ${ONEAPI_ROOT}/compiler/latest/linux/include/sycl
    )
    #target_include_directories(
    #    DPPYOneapiInterface
    #    SYSTEM PUBLIC
    #    /opt/intel/inteloneapi/compiler/latest/linux/include/sycl
    #)
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
