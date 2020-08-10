'''
    Python Data Parallel Processing Library (PyDPPL)

    PyDPPL provides a lightweight Python abstraction over DPC++/SYCL and
    OpenCL runtime objects. The DPC++ runtime wrapper objects can be
    accessed by importing dppl. The OpenCL runtime wrapper objects can be
    accessed by importing dppl.ocldrv. The library is in an early-beta
    stage of development and not yet ready for production usage.

    PyDPPL's intended usage is as a common SYCL interoperability layer for
    different Python libraries and applications. The OpenCL support inside
    PyDPPL is slated to be deprecated and then removed in future releases
    of the library.

    Currently, only a small subset of DPC++ runtime objects are exposed
    through the dppl module. The main API classes inside the dppl module are:

    Runtime:     The class stores a global SYCL queue and a stack of
                 currently activated queues. Runtime provides a special getter
                 method to retrieve the currently activated SYCL queue
                 as a Py_capsule.

                 A single global thread local instance of the Runtime class
                 is created on loading the dppl module for the first time.

    DeviceArray: A DeviceArray object encapsulates a one-dimensional
                 cl::sycl::buffer object. A DeviceArray object can be
                 created using a NumPy ndarray. The initial implementation
                 of DeviceArray follows NumPy's recommended design to create
                 a custom array container. DeviceArray does not implement
                 the __array_function__ and the __array_ufunc__ interfaces.
                 Therefore, DeviceArray does not support NumPy Universal
                 functions (ufuncs). The design decision to not support
                 ufuncs can be revisited later if we have a need for such
                 functionality. For the time being, the class is only meant
                 as a data exchange format between Python libraries that
                 use SYCL.

    Global data members:
        runtime - An instance of the Runtime class.

    Please use `pydoc dppl._oneapi_interface` to look at the current API
    for dppl.

    Please use `pydoc dppl.ocldrv` to look at the current API for dppl.ocldrv.

'''
__author__ = "Intel Corp."

from ._oneapi_interface import *
from ._version import get_versions

__version__ = get_versions()['version']
del get_versions
