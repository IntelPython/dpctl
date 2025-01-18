#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2025 Intel Corporation
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

import os
import os.path
import sys

is_venv_win32 = (
    sys.platform == "win32"
    and sys.base_exec_prefix != sys.exec_prefix
    and os.path.isfile(os.path.join(sys.exec_prefix, "pyvenv.cfg"))
)

if is_venv_win32:  # pragma: no cover
    # For virtual environments on Windows, add folder
    # with DPC++ libraries to the DLL search path gh-1745
    dll_dir = os.path.join(sys.exec_prefix, "Library", "bin")
    if os.path.isdir(dll_dir):
        os.add_dll_directory(dll_dir)

del is_venv_win32

is_linux = sys.platform.startswith("linux")

if is_linux:
    # forking is not supported by device drivers
    # Configure subprocess (used by versioneer) to
    # use SPAWN method over FORK method to enable
    # use of gdb-oneapi to debug code launched by
    # native extensions that used dpctl C/C++ API
    import subprocess

    subprocess._USE_VFORK = False
    subprocess._USE_POSIX_SPAWN = True
    # remove qualifier from this namespace
    del subprocess

del is_linux
