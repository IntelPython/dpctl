
find_package(PythonExtensions REQUIRED)
find_package(NumPy REQUIRED)
find_package(Cython REQUIRED)

# at build time create include/ directory and copy header files over
set(DPCTL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_custom_target(_build_time_create_dpctl_include ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DPCTL_INCLUDE_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DPCTL_INCLUDE_DIR}/syclinterface
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/libsyclinterface/include/* ${DPCTL_INCLUDE_DIR}/syclinterface
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/apis/include/* ${DPCTL_INCLUDE_DIR}
    DEPENDS DPCTLSyclInterface
)

set(CMAKE_INSTALL_RPATH "$ORIGIN")

function(build_dpctl_ext _trgt _src _dest)
    add_cython_target(${_trgt} ${_src} CXX OUTPUT_VAR _generated_src)
    add_library(${_trgt} MODULE ${_generated_src})
    target_include_directories(${_trgt} PRIVATE ${NumPy_INCLUDE_DIR} ${DPCTL_INCLUDE_DIR})
    add_dependencies(${_trgt} _build_time_create_dpctl_include)
    target_link_libraries(${_trgt} DPCTLSyclInterface)
    python_extension_module(${_trgt})
    get_filename_component(_name_wle ${_generated_src} NAME_WLE)
    get_filename_component(_generated_src_dir ${_generated_src} DIRECTORY)
    set(_generated_public_h "${_generated_src_dir}/${_name_wle}.h")
    set(_generated_api_h "${_generated_src_dir}/${_name_wle}_api.h")
    set(_copy_trgt "${_trgt}_copy_capi_include")
    add_custom_target(
         ${_copy_trgt} ALL
         COMMAND ${CMAKE_COMMAND}
	     -DSOURCE_FILE=${_generated_public_h}
	     -DDEST=${CMAKE_CURRENT_SOURCE_DIR}
	     -P ${CMAKE_SOURCE_DIR}/dpctl/cmake/copy_generated_headers.cmake
         COMMAND ${CMAKE_COMMAND}
	     -DSOURCE_FILE=${_generated_api_h}
	     -DDEST=${CMAKE_CURRENT_SOURCE_DIR}
	     -P ${CMAKE_SOURCE_DIR}/dpctl/cmake/copy_generated_headers.cmake
	 DEPENDS ${_trgt}
	 VERBATIM
         COMMENT "Copying Cython-generated headers to destination"
    )
    install(TARGETS ${_trgt} LIBRARY DESTINATION ${_dest})
endfunction()

file(GLOB _cython_sources *.pyx)
foreach(_cy_file ${_cython_sources})
    get_filename_component(_trgt ${_cy_file} NAME_WLE)
    build_dpctl_ext(${_trgt} ${_cy_file} "dpctl")
endforeach()

add_subdirectory(program)
add_subdirectory(memory)
add_subdirectory(tensor)
add_subdirectory(utils)
