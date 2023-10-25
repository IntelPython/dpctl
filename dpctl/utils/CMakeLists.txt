
file(GLOB _cython_sources *.pyx)
foreach(_cy_file ${_cython_sources})
    get_filename_component(_trgt ${_cy_file} NAME_WLE)
    build_dpctl_ext(${_trgt} ${_cy_file} "dpctl/utils")
endforeach()

add_custom_target(_dpctl4pybind11_header_ready
    DEPENDS
    _usmarray_copy_capi_include
    _memory_copy_capi_include
    _sycl_device_copy_capi_include
    _sycl_queue_copy_capi_include
    _sycl_context_copy_capi_include
    _sycl_event_copy_capi_include
)

set(python_module_name _device_queries)
set(_module_src ${CMAKE_CURRENT_SOURCE_DIR}/src/device_queries.cpp)
pybind11_add_module(${python_module_name} MODULE
  ${_module_src}
)
add_sycl_to_target(TARGET ${python_module_name} SOURCES ${_module_src})
if(_dpctl_sycl_targets)
    # make fat binary
    target_compile_options(
        ${python_module_name}
        PRIVATE
        -fsycl-targets=${_dpctl_sycl_targets}
    )
    target_link_options(
        ${python_module_name}
        PRIVATE
        -fsycl-targets=${_dpctl_sycl_targets}
    )
endif()
target_include_directories(${python_module_name}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
add_dependencies(${python_module_name} _dpctl4pybind11_header_ready)
install(TARGETS ${python_module_name} DESTINATION "dpctl/utils")
