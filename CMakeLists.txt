cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)
project("DP-GLUE - A minimal OpenCL and SYCL wrapper library")

if(UNIX)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}                                          \
     -Wall -Wextra -Winit-self -Wuninitialized -Wmissing-declarations          \
     -fdiagnostics-color=auto -pedantic-errors                        \
  ")
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -ggdb3 -DDEBUG ")
endif()

# We need OpenCL support
find_package(OpenCL REQUIRED)

message(OCL_INCLUDE_DIR ${OpenCL_INCLUDE_DIR})
message(OCL_LIBRARY  ${OpenCL_LIBRARY})

# Todo : Add build rules for unit tests

# Todo : Add build rules for doxygen

add_library(
    dpglue
    SHARED
    source/opencllite.cpp
)

target_include_directories(dpglue PRIVATE ${CMAKE_SOURCE_DIR}/include/)
target_include_directories(dpglue PUBLIC ${OpenCL_INCLUDE_DIR})
target_include_directories(dpglue PUBLIC ${LIBUSM_INCLUDE_DIR})
#link_directories(${LIBUSM_LIBDIR})
#target_link_libraries(dpglue PUBLIC libusm)

install(TARGETS dpglue LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/")

# Install all headers
file(GLOB NUMBA_DP_GLUE_HEADERS "${CMAKE_SOURCE_DIR}/include/*.h")
foreach(HEADER ${NUMBA_DP_GLUE_HEADERS})
    install(FILES "${HEADER}" DESTINATION include)
endforeach()
