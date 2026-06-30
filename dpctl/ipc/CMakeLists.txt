if(DPCTL_HAS_IPC_MEMORY)
    set(_cy_file ${CMAKE_CURRENT_SOURCE_DIR}/_ipc_memory.pyx)
    get_filename_component(_trgt ${_cy_file} NAME_WLE)
    build_dpctl_ext(${_trgt} ${_cy_file} "dpctl/ipc" RELATIVE_PATH "..")
    target_link_libraries(DpctlCAPI INTERFACE ${_trgt}_headers)
endif()

# Always install __init__.py (handles graceful fallback when IPC not supported)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    DESTINATION ${CMAKE_INSTALL_PREFIX}/dpctl/ipc)
