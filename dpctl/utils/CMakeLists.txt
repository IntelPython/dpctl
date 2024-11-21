
file(GLOB _cython_sources *.pyx)
foreach(_cy_file ${_cython_sources})
    get_filename_component(_trgt ${_cy_file} NAME_WLE)
    build_dpctl_ext(${_trgt} ${_cy_file} "dpctl/utils" RELATIVE_PATH "..")
endforeach()

set(_pybind11_targets)

set(python_module_name _device_queries)
set(_module_src ${CMAKE_CURRENT_SOURCE_DIR}/src/device_queries.cpp)
pybind11_add_module(${python_module_name} MODULE
  ${_module_src}
)
add_sycl_to_target(TARGET ${python_module_name} SOURCES ${_module_src})
list(APPEND _pybind11_targets ${python_module_name})


set(python_module_name _seq_order_keeper)
set(_module_src ${CMAKE_CURRENT_SOURCE_DIR}/src/order_keeper.cpp)
pybind11_add_module(${python_module_name} MODULE
  ${_module_src}
)
target_include_directories(${python_module_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_sycl_to_target(TARGET ${python_module_name} SOURCES ${_module_src})
list(APPEND _pybind11_targets ${python_module_name})

set(_linker_options "LINKER:${DPCTL_LDFLAGS}")
foreach(python_module_name ${_pybind11_targets})
    target_compile_options(${python_module_name} PRIVATE -fno-sycl-id-queries-fit-in-int)
    target_link_options(${python_module_name} PRIVATE -fsycl-device-code-split=per_kernel)
    target_include_directories(${python_module_name}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/include
        ${CMAKE_CURRENT_SOURCE_DIR}/libtensor/source/
    )
    target_link_options(${python_module_name} PRIVATE ${_linker_options})
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
            -fsycl-targets=${_dpctl_sycl_targets}
        )
        target_link_options(
            ${python_module_name}
            PRIVATE
            -fsycl-targets=${_dpctl_sycl_targets}
        )
    endif()
    # TODO: update source so they refernece individual libraries instead of
    #   dpctl4pybind11.hpp. It will allow to simplify dependency tree
    target_link_libraries(${python_module_name} PRIVATE DpctlCAPI)
    if (DPCTL_WITH_REDIST)
        set_target_properties(
            ${python_module_name}
            PROPERTIES
                INSTALL_RPATH "$ORIGIN/../../../.."
        )
    endif()
    install(TARGETS ${python_module_name} DESTINATION "dpctl/utils")
endforeach()
