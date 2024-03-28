.. _basic_concepts_intro:

oneAPI execution model
-----------------------

The Data Parallel Control (:py:mod:`dpctl`) package provides a Python runtime to access a
data-parallel computing resource or *XPU* from another Python application or
library, alleviating the need for the other Python packages to develop such a
runtime themselves. The term XPU denotes a diverse range of computing
architectures such as a CPU, GPU, FPGA, and more. They are available to programmers on a
modern heterogeneous system.

The :py:mod:`dpctl` runtime is built on top of the C++ SYCL standard as implemented in
Intel(R) oneAPI DPC++ compiler and is designed to be both vendor and architecture agnostic.
If the underlying SYCL runtime supports a type of architecture, the dpctl runtime allows
accessing that architecture from Python.

In its current form, :py:mod:`dpctl` relies on certain DPC++ extensions of the
SYCL standard. Moreover, the binary distribution of :py:mod:`dpctl` uses the proprietary
Intel(R) oneAPI DPC++ runtime bundled as part of oneAPI and is compiled to only target
Intel(R) XPU devices. :py:mod:`dpctl` supports compilation for other SYCL targets, such as
``nvptx64-nvidia-cuda`` and ``amdgcn-amd-amdhsa`` using `CodePlay plugins <codeplay_plugins_url_>`_
for oneAPI DPC++ compiler providing support for these targets.

:py:mod:`dpctl` is also compatible with the runtime of the `open-source DPC++ <os_intel_llvm_gh_url_>`_
SYCL bundle that can be compiled to support a wide range of architectures including CUDA,
AMD* ROC, and HIP*.

The user guide introduces the core features of :py:mod:`dpctl` and the underlying
concepts. The guide is meant primarily for users of the Python package. Library
and native extension developers should refer to the programmer guide.

.. _codeplay_plugins_url: https://developer.codeplay.com/products/oneapi/
.. _os_intel_llvm_gh_url: https://github.com/intel/llvm

.. toctree::
    :maxdepth: 1
    :caption: Table of Contents

    basic_concepts
    device_selection
    platforms
    devices
    queues
