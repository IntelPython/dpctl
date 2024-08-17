
file(GLOB _cython_sources *.pyx)
foreach(_cy_file ${_cython_sources})
    get_filename_component(_trgt ${_cy_file} NAME_WLE)
    build_dpctl_ext(${_trgt} ${_cy_file} "dpctl/utils")
endforeach()

set(python_module_name _device_queries)
set(_module_src ${CMAKE_CURRENT_SOURCE_DIR}/src/device_queries.cpp)
pybind11_add_module(${python_module_name} MODULE
  ${_module_src}
)
add_sycl_to_target(TARGET ${python_module_name} SOURCES ${_module_src})
if(DPCTL_GENERATE_COVERAGE)
     if(DPCTL_GENERATE_COVERAGE_FOR_PYBIND11_EXTENSIONS)
         target_compile_options(${python_module_name}
             PRIVATE -fprofile-instr-generate -fcoverage-mapping
         )
     endif()
     target_link_options(${python_module_name}
         PRIVATE -fprofile-instr-generate -fcoverage-mapping
     )
endif()
if (BUILD_DPCTL_EXT_SYCL)
    add_sycl_to_target(TARGET ${_trgt} SOURCES ${_generated_src})
    if(_dpctl_sycl_targets)
        # make fat binary
        target_compile_options(
            ${python_module_name}
            PRIVATE
            ${_dpctl_sycl_target_compile_options}
        )
        target_link_options(
            ${python_module_name}
            PRIVATE
            ${_dpctl_sycl_target_link_options}
        )
    endif()
endif()
target_link_libraries(${python_module_name} PRIVATE DpctlCAPI)
install(TARGETS ${python_module_name} DESTINATION "dpctl/utils")


set(python_module_name _seq_order_keeper)
set(_module_src ${CMAKE_CURRENT_SOURCE_DIR}/src/order_keeper.cpp)
pybind11_add_module(${python_module_name} MODULE
  ${_module_src}
)
target_include_directories(${python_module_name} PRIVATE ${CUMAKE_CURRENT_SOURCE_DIR}/src)
add_sycl_to_target(TARGET ${python_module_name} SOURCES ${_module_src})
if(DPCTL_GENERATE_COVERAGE)
     if(DPCTL_GENERATE_COVERAGE_FOR_PYBIND11_EXTENSIONS)
         target_compile_options(${python_module_name}
             PRIVATE -fprofile-instr-generate -fcoverage-mapping
         )
     endif()
     target_link_options(${python_module_name}
         PRIVATE -fprofile-instr-generate -fcoverage-mapping
     )
endif()
if(_dpctl_sycl_targets)
    # make fat binary
    target_compile_options(
        ${python_module_name}
        PRIVATE
        ${_dpctl_sycl_target_compile_options}
    )
    target_link_options(
        ${python_module_name}
        PRIVATE
        ${_dpctl_sycl_target_link_options}
    )
endif()
target_link_libraries(${python_module_name} PRIVATE DpctlCAPI)
install(TARGETS ${python_module_name} DESTINATION "dpctl/utils")
