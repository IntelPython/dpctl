.. _intro:

dpctl
-----

The Data Parallel Control (dpctl) package provides a Python runtime to access a
data-parallel computing resource or *XPU* from another Python application or
library, alleviating the need for the other Python packages to develop such a
runtime themselves. The term XPU denotes a diverse range of computing
architectures such as a CPU, GPU, FPGA, and more. They are available to programmers on a
modern heterogeneous system.

The dpctl runtime is built on top of the C++ SYCL standard and is designed to be
both vendor and architecture agnostic. If the underlying SYCL runtime supports
a type of architecture, the dpctl runtime allows accessing that architecture
from Python.

In its current form, dpctl relies on certain DPC++ extensions of the SYCL standard.
Moreover, the binary distribution of dpctl uses the proprietary Intel(R) oneAPI
DPC++ runtime bundled as part of oneAPI and supports Intel(R) XPU devices only.
However, dpctl is compatible with the runtime of the open-source DPC++ SYCL bundle
that can be compiled to support a wide range of architectures including CUDA,
AMD* ROC, and HIP*.

The user guide introduces the core features of dpctl and the underlying
concepts. The guide is meant primarily for users of the Python package. Library
and native extension developers should refer to the programmer guide.

.. toctree::
    :maxdepth: 2
    :caption: Table of Contents

    basic_concepts
    device_selection
    platforms
    devices
    queues
