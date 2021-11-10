.. _SyclQueue_api:

###############
dpctl.SyclQueue
###############

.. currentmodule:: dpctl

.. autoclass:: dpctl.SyclQueue

    .. rubric:: Attributes:

    .. autoautosummary:: dpctl.SyclQueue
        :attributes:

    .. rubric:: Private methods:

    .. autoautosummary:: dpctl.SyclQueue
        :private_methods:

    .. rubric:: Public methods:

    .. autoautosummary:: dpctl.SyclQueue
        :methods:

Detail
======

Attributes
----------

.. autoattribute:: dpctl.SyclQueue.is_in_order
.. autoattribute:: dpctl.SyclQueue.sycl_context
.. autoattribute:: dpctl.SyclQueue.sycl_device

Private methods
---------------

.. autofunction:: dpctl.SyclQueue._get_capsule


Public methods
--------------

.. autofunction:: dpctl.SyclQueue.addressof_ref
.. autofunction:: dpctl.SyclQueue.get_sycl_backend
.. autofunction:: dpctl.SyclQueue.get_sycl_context
.. autofunction:: dpctl.SyclQueue.get_sycl_device
.. autofunction:: dpctl.SyclQueue.mem_advise
.. autofunction:: dpctl.SyclQueue.memcpy
.. autofunction:: dpctl.SyclQueue.prefetch
.. autofunction:: dpctl.SyclQueue.submit
.. autofunction:: dpctl.SyclQueue.wait
