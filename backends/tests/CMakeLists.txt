string(COMPARE EQUAL "${GTEST_INCLUDE_DIR}" "" no_gtest_incl_dir)
string(COMPARE EQUAL "${GTEST_LIB_DIR}" "" no_gtest_lib_dir)

if(${no_gtest_incl_dir} OR ${no_gtest_lib_dir})
    message(WARNING
            "GTest is needed to test dpCtl's C API test cases.  Pass in \
            -DGTEST_INCLUDE_DIR and -DGTEST_LIB_DIR when you configure Cmake if\
             you wish to run dpCtl backend tests."
    )
else()
  # We need thread support for gtest
  find_package(Threads REQUIRED)

  set(CMAKE_CTEST_COMMAND ctest -V)

  # Emulate autotools like make check target to build tests
  add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})
  enable_testing()

  include_directories(${GTEST_INCLUDE_DIR})
  include_directories(../include)

  link_directories(${GTEST_LIB_DIR})

  set(DPCTL_C_API_TEST_CASES
    test_sycl_kernel_interface
    test_sycl_platform_interface
    test_sycl_program_interface
    test_sycl_queue_interface
    test_sycl_queue_manager
  )

  # Copy the spir-v input files to test build directory
  set(spirv-test-files
    multi_kernel.spv
  )
  foreach(tf ${spirv-test-files})
    file(COPY ${tf} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
  endforeach()

  foreach(TEST_CASE ${DPCTL_C_API_TEST_CASES})
    add_executable(${TEST_CASE} EXCLUDE_FROM_ALL ${TEST_CASE}.cpp)
    target_link_libraries(
      ${TEST_CASE} ${CMAKE_THREAD_LIBS_INIT} gtest DPPLSyclInterface
    )
    add_test(NAME ${TEST_CASE} COMMAND ${TEST_CASE})
    add_dependencies(check ${TEST_CASE})
  endforeach()
endif()
