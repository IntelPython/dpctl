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

set(_elementwise_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/elementwise_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/abs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/acos.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/acosh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/asin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/asinh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/atan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/atan2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/atanh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/bitwise_and.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/bitwise_invert.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/bitwise_left_shift.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/bitwise_or.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/bitwise_right_shift.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/bitwise_xor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/cbrt.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/ceil.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/conj.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/copysign.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/cos.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/cosh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/equal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/exp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/exp2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/expm1.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/floor_divide.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/floor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/greater_equal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/greater.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/hypot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/imag.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/isfinite.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/isinf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/isnan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/less_equal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/less.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/log.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/log1p.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/log2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/log10.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/logaddexp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/logical_and.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/logical_not.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/logical_or.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/logical_xor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/maximum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/minimum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/multiply.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/negative.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/not_equal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/positive.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/pow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/proj.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/real.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/remainder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/round.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/rsqrt.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/sign.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/signbit.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/sin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/sinh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/sqrt.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/square.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/subtract.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/tan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/tanh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/true_divide.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/trunc.cpp
)
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
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/repeat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/reduction_over_axis.cpp
)
list(APPEND _tensor_impl_sources
     ${_elementwise_sources}
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
  ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/reduction_over_axis.cpp
)
list(APPEND _no_fast_math_sources
     ${_elementwise_sources}
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
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/abs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/elementwise_functions/sqrt.cpp
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
