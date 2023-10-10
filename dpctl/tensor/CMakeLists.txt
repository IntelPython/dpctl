file(GLOB _cython_sources *.pyx)
foreach(_cy_file ${_cython_sources})
    get_filename_component(_trgt ${_cy_file} NAME_WLE)
    build_dpctl_ext(${_trgt} ${_cy_file} "dpctl/tensor")
    target_include_directories(${_trgt} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endforeach()

add_custom_target(_usmarray_deps SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/_slicing.pxi
    ${CMAKE_CURRENT_SOURCE_DIR}/_types.pxi
    ${CMAKE_CURRENT_SOURCE_DIR}/_stride_utils.pxi
)
add_dependencies(_usmarray _usmarray_deps)

add_custom_target(_dpctl4pybind11_deps
    DEPENDS
    _usmarray_copy_capi_include
    _memory_copy_capi_include
    _sycl_device_copy_capi_include
    _sycl_queue_copy_capi_include
    _sycl_context_copy_capi_include
    _sycl_event_copy_capi_include
)

if(WIN32)
    if (${CMAKE_VERSION} VERSION_LESS "3.23")
        # this is a work-around for target_link_options inserting option after -link option, cause
        # linker to ignore it.
        set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -fsycl-device-code-split=per_kernel")
    endif()
endif()

set(_tensor_impl_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/tensor_py.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/accumulators.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/simplify_iteration_space.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/copy_and_cast_usm_to_usm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/copy_numpy_ndarray_into_usm_ndarray.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/copy_for_reshape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/copy_for_roll.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/linear_sequences.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/integer_advanced_indexing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/boolean_advanced_indexing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/eye_ctor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/full_ctor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/triul_ctor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/where.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/boolean_reductions.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/device_support_queries.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/repeat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/reduction_over_axis.cpp
)

set(python_module_name _tensor_impl)
pybind11_add_module(${python_module_name} MODULE ${_tensor_impl_sources})
add_sycl_to_target(TARGET ${python_module_name} SOURCES ${_tensor_impl_sources})
set(_clang_prefix "")
if (WIN32)
  set(_clang_prefix "/clang:")
endif()

set(_no_fast_math_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/full_ctor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/linear_sequences.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/reduction_over_axis.cpp
)
foreach(_src_fn ${_no_fast_math_sources})
  get_source_file_property(_cmpl_options_prop ${_src_fn} COMPILE_OPTIONS)
  set(_combined_options_prop ${_cmpl_options_prop} "${_clang_prefix}-fno-fast-math")
  set_source_files_properties(
     ${_src_fn}
     PROPERTIES COMPILE_OPTIONS "${_combined_options_prop}"
  )
endforeach()
if (UNIX)
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions.cpp
    PROPERTIES COMPILE_DEFINITIONS "USE_STD_ABS_FOR_COMPLEX_TYPES;USE_STD_SQRT_FOR_COMPLEX_TYPES")
endif()
target_compile_options(${python_module_name} PRIVATE -fno-sycl-id-queries-fit-in-int)
target_link_options(${python_module_name} PRIVATE -fsycl-device-code-split=per_kernel)
if(UNIX)
    # this option is supported on Linux only
    target_link_options(${python_module_name} PRIVATE -fsycl-link-huge-device-code)
endif()
target_include_directories(${python_module_name}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/include
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/
)
set(_linker_options "LINKER:${DPCTL_LDFLAGS}")
target_link_options(${python_module_name} PRIVATE ${_linker_options})
add_dependencies(${python_module_name} _dpctl4pybind11_deps)
install(TARGETS ${python_module_name} DESTINATION "dpctl/tensor")
