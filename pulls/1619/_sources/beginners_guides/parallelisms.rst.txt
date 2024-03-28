.. _parallelism_definitions:

Types of parallelisms
=====================

Parallelism refers to an opportunity to work on multiple parts of a problem independently.

Exploiting parallelism requires capable hardware to work on more than one thing at a time,
such as GPUs or multi-core CPUs.

Two commonly encountered types of parallelism are:

* Task parallelism - problem is decomposed into independent tasks.
* Data parallelism - same task can be independently performed on different data inputs.


`Intel(R) oneAPI DPC++ <intel_oneapi_dpcpp_>`_ compiler implements SYCL standard which brings data parallelism to C++ language,
so it is apt that DPC++ stands for data-parallel C++. Please refer to open access book "`Data Parallel C++ <mastering_dpcpp_book_>`_"
by J. Rainders, et. al. for a great introduction.

.. _intel_oneapi_dpcpp: https://www.intel.com/content/www/us/en/developer/tools/oneapi/data-parallel-c-plus-plus.html
.. _mastering_dpcpp_book: https://link.springer.com/book/10.1007/978-1-4842-5574-2
