Welcome to Data-parallel Control (dpctl)'s documentation!
=========================================================

`Data-parallel control <https://github.com/IntelPython/dpctl>`_ (dpctl) is a
runtime library for Python applications and libraries to execute a compute
kernel on a device that supports a data-parallel mode of execution. Using
dpctl's API a library or an application can query a system to identify
data-parallel devices, allocate memory on those devices, and schedule execution
of compute kernels on the devices. Dpctl's role is only to facilitate the
scheduling of compute kernels, the library plays no role in
the definition of the kernels themselves. It is up to the users of dpctl to
define the kernels. As an example, the
`numba-dppy <https://intelpython.github.io/numba-dppy/latest/index.html>`_
package uses an OpenCL-like abstraction to define kernels directly in Python
and JIT compiles them to native binary. Numba-dppy then schedules the kernels on
devices using dpctl. Another example is the
`dpnp <https://intelpython.github.io/dpnp/>`_ package, a NumPy-like
library of pre-compiled kernels written in the
`SYCL language <https://sycl.readthedocs.io/en/latest/index.html>`_. Dpnp too
uses dpctl to schedule and execute the kernels it provides.

Dpctl uses SYCL as the underlying low-level runtime layer and implements
Python bindings for a subset of the runtime classes defined in
`SYCL 2020 <https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html>`_.
Currently, only Intel(R)'s `DPC++ <https://intel.ly/3wwjEsd>`_ is the only
supported SYCL implementation. Refer the User Guide and API documentation for a
comprehensive list of SYCL features exposed by dpctl.

.. todo::

    A paragraph on dpctl.tensor
