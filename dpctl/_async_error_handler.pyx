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

""" Implements the asynchronous error handler of dpctl queues and contexts,
which logs each error as a warning of the ``dpctl._async_error_handler``
logger.
"""

import logging

from ._backend cimport (  # noqa: E211
    DPCTLAsyncErrorHandlerSpec,
    error_handler_callback,
)

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

    Returns:
        logging.Logger:
            Logger that SYCL asynchronous errors are reported to.
    """
    return _logger


cdef void _async_error_handler(
    int err_code, const char *err_msg, void *user_data
) noexcept nogil:
    # the error is already in the dpctl error log, so it is fine to drop it
    # once the interpreter is finalizing
    if not dpctl_interpreter_is_usable():
        return
    with gil:
        try:
            _logger.warning(
                "Asynchronous SYCL error (code %d): %s",
                err_code,
                (<bytes>err_msg).decode("utf-8", "replace"),
            )
        except Exception:
            pass


cdef DPCTLAsyncErrorHandlerSpec _spec
_spec.callback = <error_handler_callback *>&_async_error_handler
_spec.user_data = NULL
_spec.release = NULL


cdef const DPCTLAsyncErrorHandlerSpec *get_async_error_handler() noexcept nogil:
    return &_spec
