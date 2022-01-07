cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(dpctl
    LANGUAGES CXX
    DESCRIPTION "Python interface for XPU programming"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(IntelDPCPP REQUIRED)

add_subdirectory(libsyclinterface)

file(GLOB _dpctl_capi_headers dpctl/apis/include/*.h*)
install(FILES ${_dpctl_capi_headers}
    DESTINATION dpctl/include
    COMPONENT DpctlCAPIHeaders
)

add_subdirectory(dpctl)

if (DPCTL_GENERATE_DOCS)
    add_subdirectory(docs)
endif()
