#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2024 Intel Corporation
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

from contextlib import contextmanager
from os import environ, getenv
from platform import system as sys_platform

__doc__ = (
    "Implementation module of :class:`dpctl.utils.onetrace_enabled` "
    "context manager."
)

_UNCHECKED = sys_platform() == "Linux"
del sys_platform


@contextmanager
def onetrace_enabled():
    """Enable ``onetrace`` collection for kernels executed in this context.

    .. note::
        Proper working of this utility assumes that Python interpreter
        has been launched by ``onetrace`` or ``unitrace`` tool from project
        `intel/pti-gpu <https://github.com/intel/pti-gpu>`_.

    :Example:
        Launch the Python interpreter using `onetrace` tool:

        .. code-block:: bash

            $ onetrace --conditional-collection -v -t --demangle python app.py

        Now using the context manager in the Python sessions enables
        data collection and its output for every offloaded kernel:

        .. code-block:: python

            import dpctl.tensor as dpt
            from dpctl.utils import onetrace_enabled

            # onetrace output reporting on execution of the kernel
            # should be seen, starting with "Device Timeline"
            with onetrace_enabled():
                a = dpt.arange(100, dtype='int16')

        Sample output:

        .. code-block:: text

            >>> with onetrace_enabled():
            ...     a = dpt.arange(100, dtype='int16')
            ...
            Device Timeline (queue: 0x555aee86bed0): dpctl::tensor::kernels::constructors::linear_sequence_step_kernel<short>[SIMD32 {1; 1; 1} {100; 1; 1}]<1.1> [ns] = 44034325658 (append) 44034816544 (submit) 44036296752 (start) 44036305918 (end)
            >>>
    """
    global _UNCHECKED

    if _UNCHECKED:
        _UNCHECKED = False
        if not (
            getenv("PTI_ENABLE", None) == "1"
            and "onetrace_tool" in getenv("LD_PRELOAD", "")
        ):
            import warnings

            warnings.warn(
                "It looks like Python interpreter was not started using "
                "`onetrace` utility. Using `onetrace_enabled` may have "
                "no effect. See `onetrace_enabled.__doc__` for usage.",
                RuntimeWarning,
                stacklevel=2,
            )

    _env_var_name = "PTI_ENABLE_COLLECTION"
    saved = getenv(_env_var_name, None)
    try:
        environ[_env_var_name] = "1"
        yield
    finally:
        if saved is None:
            del environ[_env_var_name]
        else:
            environ[_env_var_name] = saved
