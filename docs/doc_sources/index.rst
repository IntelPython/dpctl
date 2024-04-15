=====================
Data Parallel Control
=====================

.. _DpctlIntroduction:

Python package :py:mod:`dpctl` enables Python users to engage with multiple
compute devices commonly available in modern consumer- and server-grade
computers using industry-standard :sycl_execution_model:`SYCL execution model <>`
facilitated by Intel(R) oneAPI :dpcpp_compiler:`DPC++ compiler <>` implementing
:sycl_spec_2020:`SYCL 2020 standard <>`.

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

    .. grid-item-card:: API Reference

        API Reference contains detailed documentation of functionality provided
        in :py:mod:`dpctl` and its components.

        +++

        .. button-ref:: api_reference
            :expand:
            :color: secondary
            :click-parent:

            Access API Reference

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
   api_reference/index
   contributor_guides/index
