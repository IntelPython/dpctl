file(GLOB _cython_sources *.pyx)
foreach(_cy_file ${_cython_sources})
    get_filename_component(_trgt ${_cy_file} NAME_WLE)
    build_dpctl_ext(${_trgt} ${_cy_file} "dpctl/tensor")
    target_include_directories(${_trgt} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endforeach()

add_custom_target(_dpctl4pybind11_deps
    DEPENDS
    _usmarray_copy_capi_include
    _memory_copy_capi_include
    _sycl_device_copy_capi_include
    _sycl_queue_copy_capi_include
    _sycl_context_copy_capi_include
    _sycl_event_copy_capi_include
)

set(python_module_name _tensor_impl)
pybind11_add_module(${python_module_name} MODULE
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/tensor_py.cpp
)
target_compile_options(${python_module_name} PRIVATE -fsycl-device-code-split=per_kernel)
target_include_directories(${python_module_name}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/include
)
add_dependencies(${python_module_name} _dpctl4pybind11_deps)
install(TARGETS ${python_module_name} DESTINATION "dpctl/tensor")
