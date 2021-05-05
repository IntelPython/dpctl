cmake_minimum_required(VERSION 3.21...3.22 FATAL_ERROR)

project(dpctl
    LANGUAGES CXX
    DESCRIPTION "Python interface for XPU programming"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(IntelDPCPP REQUIRED PATHS ${CMAKE_SOURCE_DIR}/cmake NO_DEFAULT_PATH)

add_subdirectory(libsyclinterface)

file(GLOB _dpctl_capi_headers dpctl/apis/include/*.h*)
install(FILES ${_dpctl_capi_headers}
    DESTINATION dpctl/include
)

# Define CMAKE_INSTALL_xxx: LIBDIR, INCLUDEDIR
include(GNUInstallDirs)

# Fetch pybind11
include(FetchContent)

FetchContent_Declare(
  pybind11
  URL https://github.com/pybind/pybind11/archive/refs/tags/v2.9.1.tar.gz
  URL_HASH SHA256=c6160321dc98e6e1184cc791fbeadd2907bb4a0ce0e447f2ea4ff8ab56550913
)
FetchContent_MakeAvailable(pybind11)

add_subdirectory(dpctl)

if (DPCTL_GENERATE_DOCS)
    add_subdirectory(docs)
endif()
