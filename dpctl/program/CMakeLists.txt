
file(GLOB _cython_sources *.pyx)
foreach(_cy_file ${_cython_sources})
    get_filename_component(_trgt ${_cy_file} NAME_WLE)
    build_dpctl_ext(${_trgt} ${_cy_file} "dpctl/program" RELATIVE_PATH "..")
    target_link_libraries(DpctlCAPI INTERFACE ${_trgt}_headers)
endforeach()
