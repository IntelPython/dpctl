find_package(GTest REQUIRED)
# We need thread support for gtest
find_package(Threads REQUIRED)

set(CMAKE_CTEST_COMMAND ctest -V)

# Emulate autotools like make check target to build tests
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})
enable_testing()

include_directories(../include)

link_directories(${GTEST_LIB_DIR})

set(DPCTL_C_API_TEST_CASES
    test_sycl_device_interface
    test_sycl_kernel_interface
    test_sycl_platform_interface
    test_sycl_program_interface
    test_sycl_queue_interface
    test_sycl_queue_manager
    test_sycl_usm_interface
)

# Copy the spir-v input files to test build directory
set(spirv-test-files
    multi_kernel.spv
)
foreach(tf ${spirv-test-files})
    file(COPY ${tf} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endforeach()

file(GLOB_RECURSE sources ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_executable(dpctl_c_api_tests EXCLUDE_FROM_ALL ${sources})
target_link_libraries(
    dpctl_c_api_tests ${CMAKE_THREAD_LIBS_INIT} GTest::GTest DPPLSyclInterface
)
gtest_discover_tests(dpctl_c_api_tests)
add_dependencies(check dpctl_c_api_tests)

