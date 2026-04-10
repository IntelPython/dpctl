.. _beginners_guide_why_dpctl:

History of ``dpctl`` name
=========================

SYCL Execution Model
--------------------

:sycl_spec_2020:`SYCL standard <>` proposes an execution model, in which a
user controls :ref:`execution placement <user_guide_queues>` by specifying
``sycl::queue`` object as a function argument. This execution model affords
uniform API for executing code on a variety of devices addressable with SYCL:

.. code-block:: python
    :caption: Example of execution on different devices

    # Execute on CPU device
    foo(q_cpu, ...)

    # Execute on GPU device from vendor A
    foo(q_gpuA, ...)

    # Execute on GPU device from vendor B
    foo(q_gpuB, ...)

oneAPI DPC++ implementation of SYCL
-----------------------------------

Intel(R) oneAPI DPC++ compiler is an implementation of SYCL standard along
with a set of oneAPI extensions proposed for adoption into the standard.

DPC++ stands for `Data-Parallel C++ <dpcpp_book_>`_, because it brings
:ref:`data parallelism <parallelism_definitions>` to C++ language.

.. _dpcpp_book: https://link.springer.com/book/10.1007/978-1-4842-5574-2

:py:mod:`dpctl` was created out of the need to interact with DPC++ runtime
to control execution placement from LLVM as needed by :py:mod:`numba_dpex`.

The name Data Parallel ConTroL (DPCTL) stuck.

.. note::
    :py:mod:`dpctl` is not related to Open vSwitch Data Paths Control program ``osv-dpctl``
    provided by `Open vSwitch`_.

.. _Open vSwitch: https://www.openvswitch.org/

.. _parallelism_definitions:

Types of parallelisms
---------------------

Parallelism refers to an opportunity to work on multiple parts of a problem independently.

Exploiting parallelism requires capable hardware to work on more than one thing at a time,
such as GPUs or multi-core CPUs.

Two commonly encountered types of parallelism are:

* Task parallelism - problem is decomposed into independent tasks.
* Data parallelism - same task can be independently performed on different data inputs.


`Intel(R) oneAPI DPC++ <intel_oneapi_dpcpp_>`_ compiler implements SYCL standard which brings data parallelism to C++ language,
so it is appropriate that DPC++ stands for data-parallel C++. Please refer to open access book "`Data Parallel C++ <mastering_dpcpp_book_>`_"
by J. Rainders, et. al. for a great introduction.

.. _intel_oneapi_dpcpp: https://www.intel.com/content/www/us/en/developer/tools/oneapi/data-parallel-c-plus-plus.html
.. _mastering_dpcpp_book: https://link.springer.com/book/10.1007/978-1-4842-5574-2
