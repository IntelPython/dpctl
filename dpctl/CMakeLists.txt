
find_package(PythonExtensions REQUIRED)
find_package(NumPy REQUIRED)

set(CYTHON_FLAGS "-t -w ${CMAKE_SOURCE_DIR}")
find_package(Cython REQUIRED)

if(WIN32)
    string(CONCAT WARNING_FLAGS
        "-Wall "
        "-Wextra "
        "-Winit-self "
        "-Wunused-function "
        "-Wuninitialized "
        "-Wmissing-declarations "
        "-Wno-unused-parameter "
    )
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Ox ${WARNING_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Ox ${WARNING_FLAGS}")
    set(CMAKE_C_FLAGS_DEBUG
        "${CMAKE_C_FLAGS_DEBUG} ${WARNING_FLAGS} -O0 -ggdb3 -DDEBUG"
    )
    set(CMAKE_CXX_FLAGS_DEBUG
        "${CMAKE_CXX_FLAGS_DEBUG} ${WARNING_FLAGS} -O0 -ggdb3 -DDEBUG"
    )
    set(DPCTL_LDFLAGS "/NXCompat /DynamicBase")
elseif(UNIX)
    string(CONCAT WARNING_FLAGS
        "-Wall "
        "-Wextra "
        "-Winit-self "
        "-Wunused-function "
        "-Wuninitialized "
        "-Wmissing-declarations "
        "-fdiagnostics-color=auto "
    )
    string(CONCAT SDL_FLAGS
        "-fstack-protector "
        "-fstack-protector-all "
        "-fpic "
        "-fPIC "
        "-D_FORTIFY_SOURCE=2 "
        "-Wformat "
        "-Wformat-security "
        "-fno-strict-overflow "
        "-fno-delete-null-pointer-checks "
    )
    string(CONCAT CFLAGS
        "${WARNING_FLAGS}"
        "${SDL_FLAGS}"
    )
    string(CONCAT CXXFLAGS
        "${WARNING_FLAGS}"
        "${SDL_FLAGS}"
        "-fsycl "
    )
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 ${CFLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ${CXXFLAGS}")
    set(CMAKE_C_FLAGS_DEBUG
        "${CMAKE_C_FLAGS_DEBUG} ${CFLAGS} -O0 -ggdb3 -DDEBUG"
    )
    set(CMAKE_CXX_FLAGS_DEBUG
        "${CMAKE_CXX_FLAGS_DEBUG} ${CXXFLAGS} -O0 -ggdb3 -DDEBUG"
    )
    set(DPCTL_LDFLAGS "-z,noexecstack,-z,relro,-z,now")
else()
    message(FATAL_ERROR "Unsupported system.")
endif()

# at build time create include/ directory and copy header files over
set(DPCTL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_custom_target(_build_time_create_dpctl_include ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DPCTL_INCLUDE_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DPCTL_INCLUDE_DIR}/syclinterface
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DPCTL_INCLUDE_DIR}/syclinterface/Support
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DPCTL_INCLUDE_DIR}/syclinterface/Config
    DEPENDS DPCTLSyclInterface
)

set(_copied_header_files)
file(GLOB _syclinterface_h ${CMAKE_SOURCE_DIR}/libsyclinterface/include/*.h)
foreach(hf ${_syclinterface_h})
    get_filename_component(_header_name ${hf} NAME)
    set(_target_header_file ${DPCTL_INCLUDE_DIR}/syclinterface/${_header_name})
    list(APPEND _copied_header_files ${_target_header_file})
    add_custom_command(OUTPUT ${_target_header_file}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${hf} ${_target_header_file}
        DEPENDS ${hf} _build_time_create_dpctl_include
	VERBATIM
    )
endforeach()

file(GLOB _syclinterface_Support_h ${CMAKE_SOURCE_DIR}/libsyclinterface/include/Support/*.h)
foreach(hf ${_syclinterface_Support_h})
    get_filename_component(_header_name ${hf} NAME)
    set(_target_header_file ${DPCTL_INCLUDE_DIR}/syclinterface/Support/${_header_name})
    list(APPEND _copied_header_files ${_target_header_file})
    add_custom_command(OUTPUT ${_target_header_file}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${hf} ${_target_header_file}
        DEPENDS ${hf} _build_time_create_dpctl_include
    )
endforeach()

file(GLOB _syclinterface_Config_h ${CMAKE_SOURCE_DIR}/libsyclinterface/include/Config/*.h)
foreach(hf ${_syclinterface_Config_h})
    get_filename_component(_header_name ${hf} NAME)
    set(_target_header_file ${DPCTL_INCLUDE_DIR}/syclinterface/Config/${_header_name})
    list(APPEND _copied_header_files ${_target_header_file})
    add_custom_command(OUTPUT ${_target_header_file}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${hf} ${_target_header_file}
        DEPENDS ${hf} _build_time_create_dpctl_include
    )
endforeach()

file(GLOB _apis_h ${CMAKE_CURRENT_SOURCE_DIR}/apis/include/*)
foreach(hf ${_apis_h})
    get_filename_component(_header_name ${hf} NAME)
    set(_target_header_file ${DPCTL_INCLUDE_DIR}/${_header_name})
    list(APPEND _copied_header_files ${_target_header_file})
    add_custom_command(OUTPUT ${_target_header_file}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${hf} ${_target_header_file}
        DEPENDS ${hf} _build_time_create_dpctl_include
    )
endforeach()

add_custom_target(
    _build_time_create_dpctl_include_copy ALL
    DEPENDS ${_copied_header_files}
)

set(CMAKE_INSTALL_RPATH "$ORIGIN")

function(build_dpctl_ext _trgt _src _dest)
    add_cython_target(${_trgt} ${_src} CXX OUTPUT_VAR _generated_src)
    add_library(${_trgt} MODULE ${_generated_src})
    target_include_directories(${_trgt} PRIVATE ${NumPy_INCLUDE_DIR} ${DPCTL_INCLUDE_DIR})
    add_dependencies(${_trgt} _build_time_create_dpctl_include_copy)
    if (DPCTL_GENERATE_COVERAGE)
        target_compile_definitions(${_trgt} PRIVATE CYTHON_TRACE=1 CYTHON_TRACE_NOGIL=1)
        target_compile_options(${_trgt} PRIVATE -fno-sycl-use-footer)
    endif()
    target_link_libraries(${_trgt} DPCTLSyclInterface)
    target_link_options(${_trgt} PRIVATE "LINKER:${DPCTL_LDFLAGS}")
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
	     -P ${CMAKE_SOURCE_DIR}/dpctl/cmake/copy_existing.cmake
         COMMAND ${CMAKE_COMMAND}
	     -DSOURCE_FILE=${_generated_api_h}
	     -DDEST=${CMAKE_CURRENT_SOURCE_DIR}
	     -P ${CMAKE_SOURCE_DIR}/dpctl/cmake/copy_existing.cmake
	 DEPENDS ${_trgt}
	 VERBATIM
         COMMENT "Copying Cython-generated headers to dpctl"
    )
    if (DPCTL_GENERATE_COVERAGE)
         set(_copy_cxx_trgt "${_trgt}_copy_cxx")
         add_custom_target(
             ${_copy_cxx_trgt} ALL
             COMMAND ${CMAKE_COMMAND}
	         -DSOURCE_FILE=${_generated_src}
	         -DDEST=${CMAKE_CURRENT_SOURCE_DIR}
	         -P ${CMAKE_SOURCE_DIR}/dpctl/cmake/copy_existing.cmake
	     DEPENDS ${_trgt}
	     VERBATIM
             COMMENT "Copying Cython-generated source to dpctl"
         )
    endif()
    install(TARGETS ${_trgt} LIBRARY DESTINATION ${_dest})
endfunction()

file(GLOB _cython_sources *.pyx)
foreach(_cy_file ${_cython_sources})
    get_filename_component(_trgt ${_cy_file} NAME_WLE)
    build_dpctl_ext(${_trgt} ${_cy_file} "dpctl")
endforeach()

target_include_directories(_sycl_queue PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(program)
add_subdirectory(memory)
add_subdirectory(tensor)
add_subdirectory(utils)
