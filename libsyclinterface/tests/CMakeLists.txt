find_package(GTest REQUIRED)

# We need thread support for gtest
find_package(Threads REQUIRED)

# Emulate autotools like make check target to build tests
set(CMAKE_CTEST_COMMAND ctest --progress --output-on-failure -j 4)
add_custom_target(check COMMAND ${CMAKE_COMMAND} -E env DPCTL_VERBOSITY=warning ${CMAKE_CTEST_COMMAND})
enable_testing()

# Copy the spir-v input files to test build directory
set(spirv-test-files
    multi_kernel.spv
    oneD_range_kernel_inttys_fp32.spv
    oneD_range_kernel_fp64.spv
    local_accessor_kernel_inttys_fp32.spv
    local_accessor_kernel_fp64.spv
    work_group_memory_kernel_fp64.spv
    work_group_memory_kernel_inttys_fp32.spv
)

foreach(tf ${spirv-test-files})
    file(COPY ${tf} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endforeach()

file(GLOB_RECURSE
    sources ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

# Add all dpctl sources into a single executable so that we can run coverage
# analysis and generate a report.
add_executable(dpctl_c_api_tests
    EXCLUDE_FROM_ALL
    ${sources}
)
add_sycl_to_target(
    TARGET dpctl_c_api_tests
    SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_helper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sycl_context_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sycl_device_invalid_filters.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sycl_device_subdevices.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sycl_device_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sycl_device_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sycl_device_selector_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sycl_device_aspects.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sycl_event_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sycl_platform_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sycl_kernel_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sycl_kernel_bundle_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sycl_platform_invalid_filters.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sycl_queue_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sycl_queue_submit.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sycl_queue_submit_local_accessor_arg.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sycl_queue_submit_work_group_memory_arg.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sycl_queue_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sycl_usm_interface.cpp
)

if(_dpctl_sycl_targets)
    target_compile_options(
        dpctl_c_api_tests
        PRIVATE
        ${_dpctl_sycl_target_compile_options}
    )
    target_link_options(
        dpctl_c_api_tests
        PRIVATE
        ${_dpctl_sycl_target_link_options}
    )
endif()

target_link_libraries(dpctl_c_api_tests
    ${CMAKE_THREAD_LIBS_INIT}
    GTest::gtest_main
    GTest::GTest
    DPCTLSyclInterface
    ${IntelSyclCompiler_OPENCL_LIBRARY}
    # Seems to work without it...
    ${CMAKE_DL_LIBS}
)
target_include_directories(dpctl_c_api_tests
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../helper/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../include/syclinterface"
    ${LEVEL_ZERO_INCLUDE_DIR}
)

if(_dpctl_sycl_targets)
    target_compile_options(
        dpctl_c_api_tests
        PRIVATE
        ${_dpctl_sycl_target_compile_options}
    )
    target_link_options(
        dpctl_c_api_tests
        PRIVATE
        ${_dpctl_sycl_target_link_options}
    )
endif()

if(DPCTL_GENERATE_COVERAGE)
    set(object_arg "-object;")
    add_custom_target(run-c-api-tests
        COMMAND ${CMAKE_COMMAND} -E env DPCTL_VERBOSITY=warning ${CMAKE_CURRENT_BINARY_DIR}/dpctl_c_api_tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS
        DEPENDS dpctl_c_api_tests
    )
    add_custom_target(llvm-cov-show
        COMMAND ${LLVMProfdata_EXE}
        merge
        -sparse default.profraw
        -o
        dpctl.profdata
        COMMAND ${LLVMCov_EXE}
        export
        -format=lcov
        -ignore-filename-regex=/tmp/icpx*
        -instr-profile=dpctl.profdata
        "${object_arg}$<JOIN:$<TARGET_OBJECTS:DPCTLSyclInterface>,;${object_arg}>"
        > dpctl.lcov
        COMMAND ${LLVMCov_EXE}
        show
        -instr-profile=dpctl.profdata
        "${object_arg}$<JOIN:$<TARGET_OBJECTS:DPCTLSyclInterface>,;${object_arg}>"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS
        DEPENDS run-c-api-tests
    )

    add_custom_target(llvm-cov-report
        COMMAND ${LLVMProfdata_EXE}
        merge
        -sparse default.profraw
        -o
        dpctl.profdata
        COMMAND ${LLVMCov_EXE}
        export
        -format=lcov
        -ignore-filename-regex=/tmp/icpx*
        -instr-profile=dpctl.profdata
        "${object_arg}$<JOIN:$<TARGET_OBJECTS:DPCTLSyclInterface>,;${object_arg}>"
        > dpctl.lcov
        COMMAND ${LLVMCov_EXE}
        report
        -instr-profile=dpctl.profdata
        "${object_arg}$<JOIN:$<TARGET_OBJECTS:DPCTLSyclInterface>,;${object_arg}>"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS
        DEPENDS run-c-api-tests
    )

    add_custom_target(lcov-genhtml
        COMMAND ${LLVMProfdata_EXE}
        merge
        -sparse default.profraw
        -o
        dpctl.profdata
        COMMAND ${LLVMCov_EXE}
        export
        -format=lcov
        -instr-profile=dpctl.profdata
        "${object_arg}$<JOIN:$<TARGET_OBJECTS:DPCTLSyclInterface>,;${object_arg}>"
        > dpctl.lcov
        COMMAND ${GENHTML_EXE}
        ${CMAKE_CURRENT_BINARY_DIR}/dpctl.lcov
        --no-source
        --output-directory
        ${COVERAGE_OUTPUT_DIR}/dpctl-c-api-coverage
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS
        DEPENDS run-c-api-tests
    )
endif()

gtest_discover_tests(dpctl_c_api_tests)
add_dependencies(check dpctl_c_api_tests)
