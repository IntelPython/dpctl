
set(_cy_file ${CMAKE_CURRENT_SOURCE_DIR}/_memory.pyx)
get_filename_component(_trgt ${_cy_file} NAME_WLE)
build_dpctl_ext(${_trgt} ${_cy_file} "dpctl/memory" SYCL)
# _memory include _opaque_smart_ptr.hpp
target_include_directories(${_trgt} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(DpctlCAPI INTERFACE ${_trgt}_headers)
