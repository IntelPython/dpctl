.. _queues:

#####
Queue
#####

You need a queue to schedule the execution of any computation or data copying on a
device. 

The queue construction requires specifying:

* Device
* Context targeting the device
* Additional properties, such as:
  * If profiling information should be collected
  * If submitted tasks are executed in the order, in which they are submitted

The :class:`dpctl.SyclQueue` class represents a queue and abstracts the
:sycl_queue:`sycl::queue <>` SYCL runtime class.

Types of Queues
---------------

SYCL has a task-based execution model. The order, in which a SYCL runtime
executes a task on a target device, is specified by a sequence of events that
must be completed before the execution of the task is allowed. 

Submission of a task returns an event that you can use to further grow the graph of computational
tasks. A SYCL queue stores the needed data to manage the scheduling operations.

There are two types of queues: 

* **Out-of-order.** Unless specified otherwise during the constriction of a queue, a SYCL runtime
  executes tasks, which dependencies are met in an unspecified order, with the
  possibility for some of the tasks to be executed concurrently.
* **In-order.** You can specify SYCL queues to indicate that runtime must execute tasks in the
  order, in which they are submitted. In this case, tasks submitted to such a
  queue are never executed concurrently.


Creating a New Queue
--------------------

:class:`dpctl.SyclQueue(ctx, dev, property=None)` creates a new queue instance
for the given compatible context and device. 

To create the **in-order** queue, set a keyword ``parametr`` to ``in_order``

To dynamically collect task execution statistics in the returned event once the
associated task completes, set a keyword ``parametr`` to ``enable_profiling``.

.. _fig-constructing-queue-context-device-property:

.. literalinclude:: ../../../../../examples/python/sycl_queue.py
    :language: python
    :lines: 17-19, 72-89
    :caption: Constructing SyclQueue from context and device
    :linenos:

A possible output for the :ref:`fig-constructing-queue-context-device-property` example:


.. program-output:: python ../examples/python/sycl_queue.py -r create_queue_from_subdevice_multidevice_context

When a context is not specified, the :sycl_queue:`sycl::queue <>` constructor
from a device instance is called. Instead of an instance of
:class:`dpctl.SyclDevice` the argument `dev` can be a valid filter selector
string. In this case, the :sycl_queue:`sycl::queue <>` constructor with the
corresponding :oneapi_filter_selection:`sycl::ext::oneapi::filter_selector <>`
is called.

.. _fig-constructing-queue-filter-selector:

.. literalinclude:: ../../../../../examples/python/sycl_queue.py
    :language: python
    :lines: 17-19, 27-37
    :caption: Constructing SyclQueue from filter selector
    :linenos:

A possible output for the :ref:`fig-constructing-queue-filter-selector` example:

.. program-output:: python ../examples/python/sycl_queue.py -r create_queue_from_filter_selector


Profiling a Task Submitted to a Queue
-------------------------------------

The result of scheduling the execution of a task on a queue is an event. You can use
an event for several purposes: 

* Query for the status of the task execution
* Order execution of future tasks after it is completed
* Wait for execution to complete
* Сarry information to profile the task execution

The profiling information is only populated if the queue
used is created with the ``enable_profiling`` property and only becomes available
after the task execution is complete.

The :class:`dpctl.SyclTimer` class implements a Python context manager. 
You can use this context manager to collect cumulative profiling information for all the tasks submitted
to the queue of interest by functions executed within the context:

.. code-block:: python
   :caption: Example of timing execution

   import dpctl import dpctl.tensor as dpt

   q = dpctl.SyclQueue(property="enable_profiling") timer_ctx =
   dpctl.SyclTimer() with timer_ctx(q):
       X = dpt.arange(10**6, dtype=float, sycl_queue=q)

   host_dt, device_dt = timer_ctx.dt

The timer leverages :oneapi_enqueue_barrier:`oneAPI enqueue_barrier SYCL
extension <>` and submits a barrier at context entrance and a barrier at context
exit and records associated events. The elapsed device time is computed as
``e_exit.profiling_info_start - e_enter.profiling_info_end``.
