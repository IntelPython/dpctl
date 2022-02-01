#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2021 Intel Corporation
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

""" Implements developer utilities.
"""
import contextlib
import os


cdef extern from "syclinterface/dpctl_service.h":
   cdef void DPCTLService_InitLogger(const char *, const char *)
   cdef void DPCTLService_ShutdownLogger()


def _init_logger(log_dir=None):
    """Initialize logger to use given directory to save logs.

    The call has no effect if `dpctl` was not built to use logger.
    """
    cdef bytes p = b""
    cdef const char *app_name = "dpctl"
    cdef char *ld_cstr = NULL
    if log_dir:
        if not os.path.exists(log_dir):
            raise ValueError(f"Path {log_dir} does not exist")
        if isinstance(log_dir, str):
            p = bytes(log_dir, "utf-8")
        else:
            p = bytes(log_dir)
        ld_cstr = <char *>p
    DPCTLService_InitLogger(app_name, ld_cstr)


def _shutdown_logger():
    """Finalize logger.

    The call has no effect if `dpctl` was not built to use logger.
    """
    DPCTLService_ShutdownLogger()


@contextlib.contextmanager
def syclinterface_diagnostics(verbosity="warning", log_dir=None):
    """Context manager that activate verbosity of DPCTLSyclInterface
    function calls.
    """
    _allowed_verbosity = ["warning", "error"]
    if not verbosity in _allowed_verbosity:
        raise ValueError(
            f"Verbosity argument not understood. "
            f"Permitted values are {_allowed_verbosity}"
        )
    _init_logger(log_dir=log_dir)
    _saved_verbosity = os.environ.get("DPCTL_VERBOSITY", None)
    os.environ["DPCTL_VERBOSITY"] = verbosity
    try:
        yield
    finally:
        _shutdown_logger()
        if _saved_verbosity:
            os.environ["DPCTL_VERBOSITY"] = _saved_verbosity
        else:
            del os.environ["DPCTL_VERBOSITY"]
