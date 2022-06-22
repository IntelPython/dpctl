cmake_minimum_required(VERSION 3.21...3.22 FATAL_ERROR)

project(dpctl
    LANGUAGES CXX
    DESCRIPTION "Python interface for XPU programming"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Option to generate code coverage report using llvm-cov and lcov.
option(DPCTL_GENERATE_COVERAGE
    "Build dpctl with coverage instrumentation"
    OFF
)

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
    URL https://github.com/pybind/pybind11/archive/refs/tags/v2.9.2.tar.gz
    URL_HASH SHA256=6bd528c4dbe2276635dc787b6b1f2e5316cf6b49ee3e150264e455a0d68d19c1
)
FetchContent_MakeAvailable(pybind11)

add_subdirectory(dpctl)

file(GLOB _cmake_scripts ${CMAKE_SOURCE_DIR}/cmake/*.cmake)
install(FILES ${_cmake_scripts}
    DESTINATION dpctl/resources/cmake
)

if (DPCTL_GENERATE_DOCS)
    add_subdirectory(docs)
endif()
