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

import os
import os.path

if hasattr(os, "add_dll_directory"):
    # For virtual environments on Windows, add folder
    # with DPC++ libraries to the DLL search path gh-1745
    if "VIRTUAL_ENV" in os.environ:
        venv_dir = os.environ["VIRTUAL_ENV"]
        # Expect to find file per PEP-0405
        expected_file = os.path.join(venv_dir, "pyvenv.cfg")
        dll_dir = os.path.join(venv_dir, "Library", "bin")
        if os.path.isdir(dll_dir) and os.path.isfile(expected_file):
            os.add_dll_directory(dll_dir)
