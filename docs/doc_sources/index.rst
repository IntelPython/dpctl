=====================
Data Parallel Control
=====================

.. _DpctlIntroduction:

Python package :py:mod:`dpctl` enables Python users to engage with multiple
compute devices commonly available in modern consumer- and server-grade
computers using industry-standard :sycl_execution_model:`SYCL execution model <>`
facilitated by Intel(R) oneAPI :dpcpp_compiler:`DPC++ compiler <>`.

..
   :mod:`dpctl` leverages `oneAPI DPC++ compiler runtime <dpcpp_compiler>`_ to
   answer the following three questions users of heterogenous platforms ask:

   1.  What are available compute devices?
   2.  How to specify the device a computation is to be offloaded to?
   3.  How to manage sharing of data between devices and Python?

   :mod:`dpctl` implements Python classes and free functions mapping to DPC++
   entities to answer these questions.

:py:mod:`dpctl` provides a reference data-parallel implementation of
array library :py:mod:`dpctl.tensor` conforming to Python Array API specification.
The implementation adheres to a programming model affording clear control
over the compute device where array computations and memory allocations
take place.

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: Beginner Guides

        New to :py:mod:`dpctl`? Check out the Tutorials.
        They are a hands-on introduction for beginners.

        +++

        .. Tutorials contain

        .. button-ref:: beginners_guides
            :expand:
            :color: secondary
            :click-parent:

            To the beginner's guides

    .. grid-item-card:: User Guides

        The user guides are recipes for key tasks and common problems.

        +++

        .. button-ref:: user_guides
            :expand:
            :color: secondary
            :click-parent:

            To the user guides

    .. grid-item-card:: Reference Guides

        Reference guides contain detailed documentation of functionality provided
        in :py:mod:`dpctl`.

        +++

        .. button-ref:: reference_guides
            :expand:
            :color: secondary
            :click-parent:

            Access reference guides

    .. grid-item-card:: Contibutor Guides

        The contributing guidelines will suggest a process of
        contributing to :mod:`dpctl`.

        +++

        .. button-ref:: contributor_guides
            :expand:
            :color: secondary
            :click-parent:

            How can I contribute?


.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Contents:

   beginners_guides/index
   user_guides/index
   reference_guides/index
   contributor_guides/index
