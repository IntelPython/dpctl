file(GLOB _cython_sources *.pyx)
foreach(_cy_file ${_cython_sources})
    get_filename_component(_trgt ${_cy_file} NAME_WLE)
    build_dpctl_ext(${_trgt} ${_cy_file} "dpctl/tensor")
    target_include_directories(${_trgt} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endforeach()

add_custom_target(_dpctl4pybind11_deps
    DEPENDS
    _usmarray
    _memory
    _sycl_device
    _sycl_queue
    _sycl_context
    _sycl_event
)

set(python_module_name _tensor_impl)
pybind11_add_module(${python_module_name} MODULE
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/tensor_py.cpp
)
target_include_directories(${python_module_name}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/include
)
add_dependencies(${python_module_name} _dpctl4pybind11_deps)
install(TARGETS ${python_module_name} DESTINATION "dpctl/tensor")
