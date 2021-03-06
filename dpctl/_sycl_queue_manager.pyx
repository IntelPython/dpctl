#                      Data Parallel Control (dpCtl)
#
# Copyright 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# distutils: language = c++
# cython: language_level=3

from __future__ import print_function
from enum import Enum, auto
import logging
from . import backend_type, device_type
from ._backend cimport(
    _backend_type,
    _device_type,
    DPCTLPlatform_DumpInfo,
    DPCTLPlatform_GetNumNonHostPlatforms,
    DPCTLQueueMgr_GetCurrentQueue,
    DPCTLQueueMgr_GlobalQueueIsCurrent,
    DPCTLQueueMgr_PushQueue,
    DPCTLQueueMgr_PopQueue,
    DPCTLQueueMgr_SetGlobalQueue,
    DPCTLSyclQueueRef,
    DPCTLQueueMgr_GetQueueStackSize,
)
from ._sycl_context cimport SyclContext


__all__ = [
    "device_context",
    "dump",
    "get_current_backend",
    "get_current_device_type",
    "get_current_queue",
    "get_num_activated_queues",
    "get_num_platforms",
    "has_sycl_platforms",
    "is_in_device_context",
    "set_global_queue",
]

_logger = logging.getLogger(__name__)


cdef class _SyclQueueManager:
    """ Provides a SYCL queue manager interface for Python.
    """

    def _set_as_current_queue(self, arg):
        cdef SyclQueue q
        cdef DPCTLSyclQueueRef queue_ref = NULL

        if isinstance(arg, SyclQueue):
            q_obj = arg
        else:
            q_obj = SyclQueue(arg)

        q = <SyclQueue> q_obj
        queue_ref = q.get_queue_ref()
        DPCTLQueueMgr_PushQueue(queue_ref)

        return q_obj

    def _remove_current_queue(self):
        DPCTLQueueMgr_PopQueue()

    def dump(self):
        """
        Prints information about the SYCL environment.

        Currently, this function prints a list of all SYCL platforms that
        are available on the system and the list of devices for each platform.

        :Example:
            On a system with an OpenCL CPU driver, OpenCL GPU driver,
            Level Zero GPU driver, running the command. ::

            $python -c "import dpctl; dpctl.dump()"

            returns ::

                ---Platform 0::
                    Name        Intel(R) OpenCL
                    Version     OpenCL 2.1 LINUX
                    Vendor      Intel(R) Corporation
                    Profile     FULL_PROFILE
                    Backend     opencl
                    Devices     1
                ---Device 0::
                    Name                Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz
                    Driver version      2020.11.11.0.13_160000
                    Device type         cpu
                ---Platform 1::
                    Name        Intel(R) OpenCL HD Graphics
                    Version     OpenCL 3.0
                    Vendor      Intel(R) Corporation
                    Profile     FULL_PROFILE
                    Backend     opencl
                    Devices     1
                ---Device 0::
                    Name                Intel(R) Graphics Gen9 [0x3e98]
                    Driver version      20.47.18513
                    Device type         gpu
                ---Platform 2::
                    Name        Intel(R) Level-Zero
                    Version     1.0
                    Vendor      Intel(R) Corporation
                    Profile     FULL_PROFILE
                    Backend     level_zero
                    Devices     1
                ---Device 0::
                    Name                Intel(R) Graphics Gen9 [0x3e98]
                    Driver version      1.0.18513
                    Device type         gpu

        """
        DPCTLPlatform_DumpInfo()

    cpdef get_current_backend(self):
        """
        Returns the backend for the current queue as a `backend_type` enum.

        Returns:
            backend_type: The SYCL backend for the currently selected queue.
        """
        return self.get_current_queue().get_sycl_backend()

    cpdef get_current_device_type(self):
        """
        Returns current device type as a `device_type` enum.

        Returns:
            device_type: The SYCL device type for the currently selected queue.
            Possible values can be gpu, cpu, accelerator, or host.
        """
        return self.get_current_queue().get_sycl_device().get_device_type()

    cpdef SyclQueue get_current_queue(self):
        """
        Returns the currently activated SYCL queue as a new SyclQueue object.

        Returns:
            SyclQueue: If there is a currently active SYCL queue that queue
            is returned wrapped in a SyclQueue object. The SyclQueue object
            owns a copy of the currently active SYCL queue as an opaque
            `DPCTLSyclQueueRef` pointer. The pointer is freed when the SyclQueue
            is garbage collected.

        Raises:
            SyclQueueCreationError: If no currently active SYCL queue found.
        """
        return SyclQueue._create(DPCTLQueueMgr_GetCurrentQueue())

    def get_num_activated_queues(self):
        """
        Returns the number of currently activated queues for this thread.

        Whenever a program's control enters a :func:`dpctl.device_context()`
        scope, either a new SYCL queue is created or a previously created
        queue is retrieved from a cache and yielded. The queue yielded by the
        context manager is termed to be "activated". If a program creates
        multiple nested :func:`dpctl.device_context()` scopes then multiple
        queues can be activated at the same time, although only the latest
        activated queue is usable directly via calling
        :func:`dpctl.get_current_queue()`. This function returns the number of
        currently activated queues.

        Returns:
            int: The number of currently activated queues.

        """
        return DPCTLQueueMgr_GetQueueStackSize()

    def get_num_platforms(self):
        """
        Returns the number of available non-host SYCL platforms.
        *WARNING: To be depracated in the near future.*

        Returns:
            int: The number of non-host SYCL backends.
        """
        return DPCTLPlatform_GetNumNonHostPlatforms()

    def has_sycl_platforms(self):
        """
        Checks if the system has any non-host SYCL platforms. *WARNING: The
        behavior of the function may change in the future to include the host
        platform.*

        Returns:
            bool: Returns True if there is at least one non-host SYCL,
            platform, otherwise returns False.

        """
        cdef size_t num_platforms = DPCTLPlatform_GetNumNonHostPlatforms()
        if num_platforms:
            return True
        else:
            return False


    def is_in_device_context(self):
        """
        Checks if the control is inside a :func:`dpctl.device_context()` scope.

        Returns:
            bool: True if the control is within a
            :func:`dpctl.device_context()` scope, otherwise False.
        """
        cdef int inCtx = DPCTLQueueMgr_GlobalQueueIsCurrent()
        return not bool(inCtx)

    def set_global_queue(self, arg):
        """
        Sets the global queue to the SYCL queue specified explicitly,
        or created from given arguments.

        Args:
            A SyclQueue instance to be used as a global queue.
            Alternatively, a filter selector string, or a SyclDevice
            instance to be used to construct SyclQueue.

        Raises:
            SyclQueueCreationError: If a SYCL queue could not be created.
        """
        cdef SyclQueue q
        cdef DPCTLSyclQueueRef queue_ref = NULL

        if type(arg) is SyclQueue:
            q = <SyclQueue> arg
        else:
            q_obj = SyclQueue(arg)
            q = <SyclQueue> q_obj

        queue_ref = q.get_queue_ref()
        DPCTLQueueMgr_SetGlobalQueue(queue_ref)


# This private instance of the _SyclQueueManager should not be directly
# accessed outside the module.
_mgr = _SyclQueueManager()

# Global bound functions
dump                     = _mgr.dump
get_num_platforms        = _mgr.get_num_platforms
get_num_activated_queues = _mgr.get_num_activated_queues
has_sycl_platforms       = _mgr.has_sycl_platforms
set_global_queue         = _mgr.set_global_queue
is_in_device_context     = _mgr.is_in_device_context


cpdef SyclQueue get_current_queue():
    """
    Returns the currently activate SYCL queue as a new SyclQueue object.

    Returns:
        SyclQueue: If there is a currently active SYCL queue that queue
        is returned wrapped in a SyclQueue object. The SyclQueue object
        owns a copy of the currently active SYCL queue as an opaque
        `DPCTLSyclQueueRef` pointer. The pointer is freed when the SyclQueue
        is garbage collected.

    Raises:
        SyclQueueCreationError: If no currently active SYCL queue found.
    """
    return _mgr.get_current_queue()


cpdef get_current_device_type():
    """
    Returns current device type as a `device_type` enum.

    Returns:
        device_type: The SYCL device type for the currently selected queue.
        Possible values can be gpu, cpu, accelerator, or host.
    """
    return _mgr.get_current_device_type()


cpdef get_current_backend():
    """
    Returns the backend for the current queue as a `backend_type` enum.

    Returns:
        backend_type: The SYCL backend for the currently selected queue.
    """
    return _mgr.get_current_backend()


from contextlib import contextmanager


@contextmanager
def device_context(arg):
    """
    Yields a SYCL queue corresponding to the input filter string.

    This context manager "activates", *i.e.*, sets as the currently usable
    queue, the SYCL queue defined by the argument `arg`.
    The activated queue is yielded by the context manager and can also be
    accessed by any subsequent call to :func:`dpctl.get_current_queue()` inside
    the context manager's scope. The yielded queue is removed as the currently
    usable queue on exiting the context manager.

    Args:

        queue_str (str) : A string corresponding to the DPC++ filter selector.

    Yields:
        :class:`.SyclQueue`: A SYCL queue corresponding to the specified
        filter string.

    Raises:
        SyclQueueCreationError: If the SYCL queue creation failed.

    :Example:
        To create a scope within which the Level Zero GPU number 0 is active,
        a programmer needs to do the following.

        .. code-block:: python

            import dpctl
            with dpctl.device_context("level0:gpu:0"):
                pass

    """
    ctxt = None
    try:
        ctxt = _mgr._set_as_current_queue(arg)
        yield ctxt
    finally:
        # Code to release resource
        if ctxt:
            _logger.debug(
                "Removing the queue from the stack of active queues")
            _mgr._remove_current_queue()
        else:
            _logger.debug("No queue was created so nothing to do")
