.. _user_guides:

===========
User guides
===========

* Concepts relevant to :ref:`heterogeneous programming <basic_concepts>`
* Definition and explanation of :ref:`basic concepts <oneapi_programming_model_intro>` involved in SYCL execution model

* Overview of array library :py:mod:`dpctl.tensor` conforming to Python array API specification
* Writing custom operations on :py:class:`dpctl.tensor.usm_ndarray` container
   - Write kernels using :py:mod:`numba_dpex`
   - Write Python extensions in SYCL using Intel(R) oneAPI DPC++ compiler and :py:mod:`dpctl`

* :ref:`Protocol <dpctl_tensor_dlpack_support>` for exchanging USM allocations using DLPack

..
   :mod:`dpctl` leverages `oneAPI DPC++ compiler runtime <dpcpp_compiler>`_ to
   answer the following three questions users of heterogenous platforms ask:

   1.  What are available compute devices?
   2.  How to specify the device a computation is to be offloaded to?
   3.  How to manage sharing of data between devices and Python?

   :mod:`dpctl` implements Python classes and free functions mapping to DPC++
   entities to answer these questions.


.. toctree::
   :hidden:

   intro
   basic_concepts
   license
   dlpack
   environment_variables
