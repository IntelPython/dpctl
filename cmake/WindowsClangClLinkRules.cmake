# CMake links clang-cl targets by calling the linker directly, so link options
# such as -fsycl never reach the compiler driver. Link through the driver
# instead, like Platform/Windows-IntelLLVM does for icx, passing linker flags
# with -Xlinker (clang-cl drops -Wl, arguments).

if(CMAKE_GENERATOR MATCHES "Visual Studio")
  return()
endif()

foreach(_lang C CXX)
  if(NOT CMAKE_${_lang}_COMPILER_ID STREQUAL "Clang"
     OR NOT CMAKE_${_lang}_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"
     OR CMAKE_${_lang}_LINK_MODE STREQUAL "DRIVER")
    continue()
  endif()

  if(NOT _dpctl_clang_cl_linker_flags_wrapped)
    set(_dpctl_clang_cl_linker_flags_wrapped TRUE)
    foreach(_t EXE SHARED MODULE)
      foreach(_cfg "" _DEBUG _RELEASE _RELWITHDEBINFO _MINSIZEREL)
        set(_var CMAKE_${_t}_LINKER_FLAGS${_cfg}_INIT)
        separate_arguments(_flags WINDOWS_COMMAND "${${_var}}")
        list(TRANSFORM _flags PREPEND "-Xlinker ")
        list(JOIN _flags " " ${_var})
      endforeach()
    endforeach()
    set(CMAKE_LIBRARY_PATH_FLAG "-Xlinker /LIBPATH:")
    set(CMAKE_LINK_DEF_FILE_FLAG "-Xlinker /DEF:")
  endif()

  set(CMAKE_${_lang}_LINK_MODE DRIVER)
  set(CMAKE_${_lang}_LINKER_WRAPPER_FLAG "-Xlinker" " ")
  unset(CMAKE_${_lang}_LINKER_WRAPPER_FLAG_SEP)
  set(CMAKE_${_lang}_LINK_DEF_FILE_FLAG "-Xlinker /DEF:")
  set(CMAKE_${_lang}_CREATE_WIN32_EXE "-Xlinker /subsystem:windows")
  set(CMAKE_${_lang}_CREATE_CONSOLE_EXE "-Xlinker /subsystem:console")

  set(CMAKE_${_lang}_LINK_EXECUTABLE
    "${_CMAKE_VS_LINK_EXE}<CMAKE_${_lang}_COMPILER> ${CMAKE_CL_NOLOGO} <OBJECTS> ${CMAKE_START_TEMP_FILE} <LINK_FLAGS> <LINK_LIBRARIES> -link /out:<TARGET> /implib:<TARGET_IMPLIB> /pdb:<TARGET_PDB> /version:<TARGET_VERSION_MAJOR>.<TARGET_VERSION_MINOR>${_PLATFORM_LINK_FLAGS} ${CMAKE_END_TEMP_FILE}")
  set(CMAKE_${_lang}_CREATE_SHARED_LIBRARY
    "${_CMAKE_VS_LINK_DLL}<CMAKE_${_lang}_COMPILER> ${CMAKE_CL_NOLOGO} <OBJECTS> ${CMAKE_START_TEMP_FILE} -LD <LINK_FLAGS> <LINK_LIBRARIES> -link /out:<TARGET> /implib:<TARGET_IMPLIB> /pdb:<TARGET_PDB> /version:<TARGET_VERSION_MAJOR>.<TARGET_VERSION_MINOR>${_PLATFORM_LINK_FLAGS} ${CMAKE_END_TEMP_FILE}")
  set(CMAKE_${_lang}_CREATE_SHARED_MODULE "${CMAKE_${_lang}_CREATE_SHARED_LIBRARY}")
endforeach()

unset(_lang)
unset(_t)
unset(_cfg)
unset(_var)
unset(_flags)
