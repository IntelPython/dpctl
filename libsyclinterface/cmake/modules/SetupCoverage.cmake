
function(setup_coverage_generation)
    # check if lcov is available
    find_package(Lcov REQUIRED)
    # check if llvm-cov version 11 is available
    find_package(LLVMCov 11 REQUIRED)
    # check if llvm-profdata is available
    find_package(LLVMProfdata REQUIRED)

    string(CONCAT PROFILE_FLAGS
        "-fprofile-instr-generate "
        "-fcoverage-mapping "
        "-fno-sycl-use-footer "
#        "-save-temps=obj "
    )

    # Add profiling flags
    set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} ${PROFILE_FLAGS}"
        PARENT_SCOPE
    )
endfunction(setup_coverage_generation)
