#                      Data Parallel Control (dpctl)
#
# Copyright 2026 Intel Corporation
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

# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True
# cython: freethreading_compatible = True

""" Implements the default SYCL asynchronous error handler used by dpctl.

SYCL reports asynchronous errors by invoking an async handler, which the SYCL
runtime may call on a thread of its choosing, at any point where errors are
flushed (typically ``queue::wait()``, or queue/context destruction).

The handler installed here logs the error through the standard :mod:`logging`
machinery. It deliberately does not raise: an exception raised on a
SYCL-internal thread cannot be delivered to the code that submitted the work,
and would terminate the process.
"""

import logging

from ._backend cimport error_handler_callback  # noqa: E211

__all__ = [
    "_get_async_error_logger",
]

_logger = logging.getLogger(__name__)


cdef extern from *:
    """
    #include "Python.h"

    static int dpctl_interpreter_is_usable(void)
    {
        if (!Py_IsInitialized()) {
            return 0;
        }
    #if PY_VERSION_HEX < 0x030D0000
        return (_Py_IsFinalizing()) ? 0 : 1;
    #else
        return (Py_IsFinalizing()) ? 0 : 1;
    #endif
    }
    """
    int dpctl_interpreter_is_usable() nogil


def _get_async_error_logger():
    """_get_async_error_logger()

    Returns the :class:`logging.Logger` instance used to report SYCL
    asynchronous errors. Provided so that applications and tests can attach
    handlers to it, or adjust its level.

    Returns:
        logging.Logger:
            Logger used by the default asynchronous error handler.
    """
    return _logger


cdef void _async_error_handler(
    int err_code, const char *err_msg
) noexcept nogil:
    """Default async error handler, called by libDPCTLSyclInterface.

    Must never raise, and must tolerate being called while the interpreter is
    shutting down, since queues and contexts may outlive it.
    """
    if not dpctl_interpreter_is_usable():
        # The interpreter is gone or going away; touching Python objects now
        # would crash. The error has already been reported by the C++
        # error_handler, so there is nothing further to do.
        return
    with gil:
        try:
            _logger.warning(
                "Asynchronous SYCL error (code %d): %s",
                err_code,
                (<bytes>err_msg).decode("utf-8", "replace")
                if err_msg is not NULL else "<no message>",
            )
        except Exception:
            # Logging must not turn an asynchronous error into a crash. This
            # can legitimately happen late in interpreter shutdown, when the
            # logging module has already been torn down.
            pass


cdef error_handler_callback * get_async_error_handler() noexcept nogil:
    """Returns a pointer to the default async error handler.

    This accessor exists because taking the address of a ``cdef`` function
    defined in another Cython module yields the address of the imported
    function *pointer*, not of the function itself. Resolving ``&`` here, in
    the module that defines the handler, gives callers the correct address.
    """
    return <error_handler_callback *>&_async_error_handler
