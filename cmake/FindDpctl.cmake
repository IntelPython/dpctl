#.rst:
#
# Find the include directory for ``dpctl_capi.h``, ``dpctl4pybind11.hpp``.
#
# This module sets the following variables:
#
# ``Dpctl_FOUND``
#   True if DPCTL was found.
# ``Dpctl_INCLUDE_DIRS``
#   The include directories needed to use Dpctl.
# ``Dpctl_VERSION``
#   The version of DPCTL found.
#
# The module will also explicitly define one cache variable:
#
# ``Dpctl_INCLUDE_DIR``
#

if(NOT Dpctl_FOUND)
  set(_find_extra_args)
  if(Dpctl_FIND_REQUIRED)
    list(APPEND _find_extra_args REQUIRED)
  endif()
  if(Dpctl_FIND_QUIET)
    list(APPEND _find_extra_args QUIET)
  endif()
  find_package(PythonInterp ${_find_extra_args})
  find_package(PythonLibs ${_find_extra_args})

  if(PYTHON_EXECUTABLE)
    execute_process(COMMAND "${PYTHON_EXECUTABLE}"
      -c "import dpctl; print(dpctl.get_include())"
      OUTPUT_VARIABLE _dpctl_include_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      )
    execute_process(COMMAND "${PYTHON_EXECUTABLE}"
      -c "import dpctl; print(dpctl.__version__)"
      OUTPUT_VARIABLE Dpctl_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      )

  endif()
endif()

find_path(Dpctl_INCLUDE_DIR
  dpctl_capi.h dpctl4pybind11.hpp dpctl_sycl_interface.h
  PATHS "${_dpctl_include_dir}" "${PYTHON_INCLUDE_DIR}"
  PATH_SUFFIXES dpctl/include
  )

set(Dpctl_INCLUDE_DIRS ${Dpctl_INCLUDE_DIR})

# handle the QUIETLY and REQUIRED arguments and set Dpctl_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Dpctl
                                  REQUIRED_VARS
                                    Dpctl_INCLUDE_DIR
                                  VERSION_VAR Dpctl_VERSION
                                  )

mark_as_advanced(Dpctl_INCLUDE_DIR)
