.. _dpctl_cmake_support:

CMake support
=============

:py:mod:`dpctl` comes with configuration file `dpctl-config.cmake` which is installed
on the `standard search path CMake uses to search for packages <cmake_find_package_search_paths_>`_.

To build your extension that leverages :py:mod:`dpctl` include the following line in your cmake script:

.. code-block:: cmake

    find_package("Dpctl" REQUIRED)

The "Dpctl" package exports the following variables:

.. list-table::

    * - ``Dpctl_INCLUDE_DIR``
      - Location of headers for using :py:mod:`dpctl` in extensions

    * - ``Dpctl_TENSOR_INCLUDE_DIR``
      - Location of headers implementing SYCL kernels powering :py:mod:`dpctl.tensor`

An example of "CMakeLists.txt" file for building an extension could be found in
`examples/pybind11 <examples_pybind11_>`_ folder in the project repository, or
in `sample-data-parallel-extensions <sample_dp_exts_>`_ repository.

.. _cmake_find_package_search_paths: https://cmake.org/cmake/help/latest/command/find_package.html
.. _examples_pybind11: https://github.com/IntelPython/dpctl/blob/master/examples/pybind11
.. _sample_dp_exts: https://github.com/IntelPython/sample-data-parallel-extensions
