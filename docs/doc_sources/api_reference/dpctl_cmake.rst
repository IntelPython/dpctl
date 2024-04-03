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

.. _cmake_find_package_search_paths: https://cmake.org/cmake/help/latest/command/find_package.html
