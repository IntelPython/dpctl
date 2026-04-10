.. _user_guides:

===========
User guides
===========

* Concepts relevant to :ref:`heterogeneous programming <basic_concepts>`.

* oneAPI :ref:`execution model <oneapi_programming_model_intro>` in Python

* :ref:`Protocol <dpctl_tensor_dlpack_support>` for exchanging USM allocations using DLPack

* :ref:`Environment variables <user_guides_env_variables>` affecting :mod:`dpctl`


.. Further topics:
   ## Where dpctl.tensor goes beyond array API

      - support for float16
      - support for out= keyword
      - Support for combining basic and advanced indexing
      - Additional API functions:
            - dpt.place
            - dpt.put
            - dpt.extract
            - Extended dpt.take
            - dpt.cbrt
            - dpt.rsqrt
            - dpt.reciprocal
            - dpt.cumulative_logsumexp
            - dpt.reduce_hypot
            - dpt.allclose
         - Mutability tutorial
            - 0D arrays, no scalars
            - array is mutable, pitfalls and best practices

   ## Using tools to understand performance

      - Getting unitrace
      - Using it to check GPU activity
      - Using it to collect tracing information
      - Using VTune
      - Using ITT API to zoom in on specific portion of your program

   ## Building DPC++ based Python extension with dpctl

   - Compatibility with system compiler (Intel LLVM is compatible with GCC runtime/VS runtime)
   - Simple example
   - List examples from dpctl
         - Document each native extension example

.. toctree::
   :hidden:

   basic_concepts
   execution_model
   dlpack
   environment_variables
