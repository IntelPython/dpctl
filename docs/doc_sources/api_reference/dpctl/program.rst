.. _dpctl_program_pyapi:

:py:mod:`dpctl.program`
=======================

:py:mod:`dpctl.program` provides a way to create a SYCL kernel
from either an OpenCL program source code represented as a string
or a SPIR-V binary file.

It implements creation of interoperability
``sycl::kernel_bundle<sycl::bundle_state_executable>`` (a collection of kernels),
as well as creation of individual ``sycl::kernel``, suitable for submission for
execution via :py:meth:`dpctl.SyclQueue.submit`.

.. py:module:: dpctl.program

.. currentmodule:: dpctl.program

.. autosummary::
    :toctree: generated
    :nosignatures:

    create_program_from_source
    create_program_from_spirv

.. autosummary::
    :toctree: generated
    :nosignatures:

    SyclProgram
    SyclKernel

.. autosummary::
    :toctree: generated
    :nosignatures:

    SyclProgramCompilationError
