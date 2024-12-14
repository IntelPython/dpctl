cmake_minimum_required(VERSION 3.22...3.27 FATAL_ERROR)

project(example_cython_blackscholes_usm VERSION 0.1 LANGUAGES CXX
  DESCRIPTION "Example of Cython extension calling SYCL routines")
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

# -t is to only Cythonize sources with timestamps newer than existing CXX files (if present)
# -w is to set working directory (and correctly set __pyx_f[] array of filenames)
set(CYTHON_FLAGS "-t -w \"${CMAKE_SOURCE_DIR}\"")
find_package(Cython REQUIRED)

find_package(TBB REQUIRED)

set(MKL_ARCH "intel64")
set(MKL_LINK "dynamic")
set(MKL_THREADING "tbb_thread")
set(MKL_INTERFACE "ilp64")
find_package(MKL REQUIRED)

set(py_module_name _blackscholes_usm)

set(_cy_source blackscholes/_blackscholes_usm.pyx)
add_cython_target(${py_module_name} ${_cy_source} CXX OUTPUT_VAR _generated_cy_src)
Python_add_library(${py_module_name} MODULE WITH_SOABI ${_generated_cy_src})
add_sycl_to_target(TARGET ${py_module_name} SOURCES ${_generated_cy_src})
target_compile_definitions(${py_module_name} PRIVATE -DMKL_ILP64)
target_include_directories(${py_module_name} PUBLIC src ${Dpctl_INCLUDE_DIRS})
target_link_libraries(${py_module_name} PRIVATE MKL::MKL_SYCL Python::NumPy)

install(TARGETS ${py_module_name} DESTINATION blackscholes)

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
