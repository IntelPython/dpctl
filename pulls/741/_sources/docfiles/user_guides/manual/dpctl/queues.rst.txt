.. _queues:

#####
Queue
#####

A queue is used to specify a device and a specific set of features of that
device on which a task is scheduled. The :class:`dpctl.SyclQueue` class
represents a queue and abstracts the :sycl_queue:`sycl::queue <>` SYCL runtime
class.

Types of Queues
---------------

Creating a New Queue
--------------------

Profiling a Task Submitted to a Queue
-------------------------------------
