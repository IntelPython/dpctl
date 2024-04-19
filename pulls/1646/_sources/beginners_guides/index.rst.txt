.. _beginners_guides:

=================
Beginner's guides
=================

Introduction
------------

:mod:`dpctl` brings the standard-based execution model to program a heterogeneous system
to Python through invocations of oneAPI-based native libraries, their Python interfaces,
or by using DPC++-based Python native extensions built using :mod:`dpctl` integration with
Python native extension generators.

The :py:mod:`dpctl` runtime is built on top of the C++ SYCL-2020 standard as implemented in
`Intel(R) oneAPI DPC++ compiler <dpcpp_compiler>`_ and is designed to be both vendor and
architecture agnostic.

Installation
------------

* :ref:`Installing <dpctl_installation>` :mod:`dpctl`
* Setting up drivers

Working with devices
--------------------

* :ref:`Managing devices <beginners_guide_managing_devices>`

Introduction to array library
-----------------------------

* :ref:`Getting started <beginners_guide_tensor_intro>` with :mod:`dpctl.tensor`

Miscellaneous
-------------

* History of ``"dpctl"`` :ref:`name <beginners_guide_why_dpctl>`
* Frequently asked questions

.. toctree::
    :hidden:

    installation
    managing_devices
    tensor_intro
    misc
