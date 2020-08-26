string(COMPARE EQUAL "${GTEST_INCLUDE_DIR}" "" no_gtest_incl_dir)
string(COMPARE EQUAL "${GTEST_LIB_DIR}" "" no_gtest_lib_dir)

if(${no_gtest_incl_dir} OR ${no_gtest_lib_dir})
    message(WARNING
            "GTest is needed to test PyDPPL's backend test cases.  Pass in \
            -DGTEST_INCLUDE_DIR and -DGTEST_LIB_DIR when you configure Cmake if\
             you wish to run PyDPPL backend tests."
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

  set(PYDPPL_BACKEND_TEST_CASES
    test_sycl_queue_manager
  )

  foreach(TEST_CASE ${PYDPPL_BACKEND_TEST_CASES})
    add_executable(${TEST_CASE} EXCLUDE_FROM_ALL ${TEST_CASE}.cpp)
    target_link_libraries(
      ${TEST_CASE} ${CMAKE_THREAD_LIBS_INIT} gtest DPPLSyclInterface
    )
    add_test(NAME ${TEST_CASE} COMMAND ${TEST_CASE})
    add_dependencies(check ${TEST_CASE})
  endforeach()
endif()
