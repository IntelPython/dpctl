.. _dpctl_pybind11_api:

pybind11 API
============

:py:mod:`dpctl` builds on top of :ref:`C-API <dpctl_capi>` to integrate with
`pybind11 <pybind11_url>`_ Python extension generator.

.. code-block:: c++
    :caption: Example of pybind11 extension using :py:mod:`dpctl` types

    // filename: _example.cpp
    #include <string>
    #include <pybind11/pybind11.h>
    #include <pybind11/stl.h>
    #include "dpctl4pybind11.hpp"

    std::string get_device_name(const sycl::device &dev) {
        return dev.get_info<sycl::info::device::name>();
    }

    PYBIND11_MODULE(_example, m) {
        m.def("get_device_name", &get_device_name);
    }

The extension should be compiled using Intel(R) oneAPI DPC++ compiler:

.. code-block:: bash

    icpx -fsycl $(python -m pybind11 --includes) $(python -m dpctl  --library) \
        _example.cpp -fPIC -shared -o _example.so

We can now use it from Python:

.. code-block:: python

    import _example
    import dpctl

    dev = dpctl.select_default_device()
    # invoke function in the extension
    print(_example.get_device_name(dev))
    # compare with value of corresponding built-in
    # device descriptor
    print(dev.name)

.. _pybind11_url: https://pybind11.readthedocs.io/
