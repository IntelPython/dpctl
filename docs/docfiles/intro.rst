Data-Parallel Control - The Library That Controls XPUs
=======================================================

Data Parallel Control ``dpctl`` is the Python library that controls multiple devices of a platform, features classes for
unified shared memory (USM) management, and implements tensor array API on top of it. It is a foundational part of
a larger family of libraries and tools for Data Parallel Python (DPPY) aimed to program XPUs the same way as CPUs.

The ``dpctl`` library is built upon `SYCL standard`<https://www.khronos.org/sycl/> and implements a subset of
`runtime classes specifications`<https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#_sycl_runtime_classes>,
which allow users to query SYCL platforms, discover and represent SYCL devices, and construct SYCL queues for execution
of data-parallel code.

The library also assists authors of Python native extensions written in C, Cython, or pybind11 to access objects
representing devices, queues, memory, and tensor array APIs.