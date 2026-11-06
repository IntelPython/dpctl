.. _dpctl_pybind11_api:

pybind11 API
============

:py:mod:`dpctl` builds on top of :ref:`C API <dpctl_capi>` to integrate with the
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

    PYBIND11_MODULE(_example, m, py::mod_gil_not_used()) {
        m.def("get_device_name", &get_device_name);
    }

The extension should be compiled using the Intel(R) oneAPI DPC++ compiler:

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

Type caster for ``bfloat16``
----------------------------

When the DPC++ compiler used to compile an extension provides the
``sycl_ext_oneapi_bfloat16`` extension, ``dpctl4pybind11.hpp`` defines the macro
``DPCTL_HAS_BFLOAT16`` and a type caster connecting
``sycl::ext::oneapi::bfloat16`` to Python. A function taking such an argument
accepts any Python object convertible to a :py:class:`float`, such as an
``ml_dtypes.bfloat16`` scalar. Its value is rounded to C++ ``float`` and then
to ``bfloat16``, so it is not always the nearest ``bfloat16``. A returned
``bfloat16`` becomes a Python :py:class:`float`.

Since the caster is only defined when the extension is available, code that has
to build with compilers that do not provide it should guard its use:

.. code-block:: c++

    #ifdef DPCTL_HAS_BFLOAT16
    sycl::ext::oneapi::bfloat16 round_trip(sycl::ext::oneapi::bfloat16 value) {
        return value;
    }
    #endif

.. _pybind11_url: https://pybind11.readthedocs.io/
