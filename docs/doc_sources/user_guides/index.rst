.. _user_guides:

===========
User guides
===========

* Definition and explanation of :ref:`basic concepts <basic_concepts_intro>` involved in SYCL execution model

* Overview of array library :py:mod:`dpctl.tensor` conforming to Python array API specification
* Writing custom operations on :py:class:`dpctl.tensor.usm_ndarray` container
   - Write kernels using :py:mod:`numba_dpex`
   - Write Python extensions in SYCL using Intel(R) oneAPI DPC++ compiler and :py:mod:`dpctl`

* :ref:`Protocol <dpctl_tensor_dlpack_support>` for exchanging USM allocations using DLPack

.. toctree::
   :hidden:

   intro
   license
   dlpack
