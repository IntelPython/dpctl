#.rst:
#
# Find the include directory for ``dpctl_capi.h``, ``dpctl4pybind11.hpp``.
#
# This module sets the following variables:
#
# ``Dpctl_FOUND``
#   True if DPCTL was found.
# ``Dpctl_INCLUDE_DIR``
#   The include directory needed to use dpctl.
# ``Dpctl_TENSOR_INCLUDE_DIR``
#   The include directory for tensor kernels implementation.
# ``Dpctl_VERSION``
#   The version of dpctl found.
#
# The module will also explicitly define two cache variables:
#
# ``Dpctl_INCLUDE_DIR``
# ``Dpctl_TENSOR_INCLUDE_DIR``
#

if(NOT Dpctl_FOUND)
  find_package(Python 3.9 REQUIRED
    COMPONENTS Interpreter Development.Module)

  if(Python_EXECUTABLE)
    execute_process(COMMAND "${Python_EXECUTABLE}"
      -m dpctl --include-dir
      OUTPUT_VARIABLE _dpctl_include_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      )
    execute_process(COMMAND "${Python_EXECUTABLE}"
      -c "import dpctl; print(dpctl.__version__)"
      OUTPUT_VARIABLE Dpctl_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      )
  endif()
endif()

find_path(Dpctl_INCLUDE_DIR
  dpctl_capi.h dpctl4pybind11.hpp dpctl_sycl_interface.h
  PATHS "${_dpctl_include_dir}" "${Python_INCLUDE_DIRS}"
  PATH_SUFFIXES dpctl/include
  )
get_filename_component(_dpctl_dir ${_dpctl_include_dir} DIRECTORY)

find_path(Dpctl_TENSOR_INCLUDE_DIR
  kernels utils
  PATHS "${_dpctl_dir}/tensor/libtensor/include"
  )

set(Dpctl_INCLUDE_DIRS ${Dpctl_INCLUDE_DIR})

# handle the QUIETLY and REQUIRED arguments and set Dpctl_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Dpctl
                                  REQUIRED_VARS
                                    Dpctl_INCLUDE_DIR Dpctl_TENSOR_INCLUDE_DIR
                                  VERSION_VAR Dpctl_VERSION
                                  )

mark_as_advanced(Dpctl_INCLUDE_DIR)
mark_as_advanced(Dpctl_TENSOR_INCLUDE_DIR)
