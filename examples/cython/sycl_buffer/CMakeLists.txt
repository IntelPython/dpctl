cmake_minimum_required(VERSION 3.22...3.27 FATAL_ERROR)

project(example_cython_syclbuffer VERSION 0.1 LANGUAGES CXX
  DESCRIPTION "Example of Cython extension to work on host allocated NumPy array using SYCL buffers and SYCL functions.")
set(DPCTL_CMAKE_MODULES_PATH "${CMAKE_SOURCE_DIR}/../../../cmake")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${DPCTL_CMAKE_MODULES_PATH})

find_package(IntelSYCL REQUIRED PATHS ${DPCTL_CMAKE_MODULES_PATH} NO_DEFAULT_PATH)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define CMAKE_INSTALL_xxx: LIBDIR, INCLUDEDIR
include(GNUInstallDirs)

find_package(Python REQUIRED COMPONENTS Development.Module NumPy)
find_package(Dpctl REQUIRED)

# -w is to set working directory (and correctly set __pyx_f[] array of filenames)
set(CYTHON_FLAGS "-w \"${CMAKE_SOURCE_DIR}\" --module-name _syclbuffer")
find_package(Cython REQUIRED)

set(py_module_name _syclbuffer)

set(_cy_source syclbuffer/_buffer_example.pyx)
add_cython_target(${py_module_name} ${_cy_source} CXX OUTPUT_VAR _generated_cy_src)
Python_add_library(${py_module_name} MODULE WITH_SOABI ${_generated_cy_src})
add_sycl_to_target(TARGET ${py_module_name} SOURCES ${_generated_cy_src})
target_include_directories(${py_module_name} PUBLIC src ${Dpctl_INCLUDE_DIRS})
target_link_libraries(${py_module_name} PRIVATE Python::NumPy)

install(TARGETS ${py_module_name} DESTINATION syclbuffer)

foreach(_src_fn ${_sources})
    get_source_file_property(_compile_options ${_src_fn} COMPILE_OPTIONS)
    set(_combined_options ${_compile_options} "-O3")
    set_source_files_properties(${_src_fn}
        PROPERTIES
        COMPILE_OPTIONS "${_combined_options}"
    )
endforeach()
target_link_options(${py_module_name} PRIVATE -fsycl-device-code-split=per_kernel)

set(ignoreMe "${SKBUILD}")
