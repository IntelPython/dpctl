.. _queues:

#####
Queue
#####

A queue is needed to schedule execution of any computation or data copying on a
device. Queue construction requires specifying a device and a context targeting
that device as well as additional properties, such as whether profiling
information should be collected or whether submitted tasks are executed in the
order in which they were submitted.

The :class:`dpctl.SyclQueue` class represents a queue and abstracts the
:sycl_queue:`sycl::queue <>` SYCL runtime class.

Types of Queues
---------------

SYCL has a task-based execution model. The order in which a SYCL runtime
executes a task on a target device is specified by a sequence of events which
must be complete before execution of the task is allowed. Submission of a task
returns an event which can be used to further grow the graph of computational
tasks. A SYCL queue stores the needed data to manage the scheduling operations.

Unless specified otherwise during constriction of a queue, a SYCL runtime
executes tasks whose dependencies were met in an unspecified order, with
possibility for some of the tasks to be execute concurrently. Such queues are
said to be 'out-of-order'.

SYCL queues can be specified to indicate that runtime must execute tasks in the
order in which they were submitted. In this case, tasks submitted to such a
queue, called 'in-order' queues, are never executed concurrently.

Creating a New Queue
--------------------

:class:`dpctl.SyclQueue(ctx, dev, property=None)` creates a new queue instance
for the given compatible context and device. Keyword parameter `property` can be
set to `"in_order"` to create an 'in-order' queue and to `"enable_profiling"` to
dynamically collect task execution statistics in the returned event once the
associated task completes.

.. _fig-constructing-queue-context-device-property:

.. literalinclude:: ../../../../../examples/python/sycl_queue.py
    :language: python
    :lines: 17-19, 72-89
    :caption: Constructing SyclQueue from context and device
    :linenos:

A possible output for the example
:ref:`fig-constructing-queue-context-device-property` may be:

.. program-output:: python ../examples/python/sycl_queue.py -r create_queue_from_subdevice_multidevice_context

When a context is not specified the :sycl_queue:`sycl::queue <>` constructor
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

A possible output for the example :ref:`fig-constructing-queue-filter-selector`
may be:

.. program-output:: python ../examples/python/sycl_queue.py -r create_queue_from_filter_selector


Profiling a Task Submitted to a Queue
-------------------------------------

The result of scheduling execution of a task on a queue is an event. An event
has several uses: it can be queried for the status of the task execution, it can
be used to order execution of the future tasks after it is complete, it can be
used to wait for execution to complete, and it can carry information to profile
of the task execution. The profiling information is only populated if the queue
used was created with the "enable_profiling" property and only becomes available
after the task execution is complete.

The class :class:`dpctl.SyclTimer` implements a Python context manager that can
be used to collect cumulative profiling information for all the tasks submitted
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
