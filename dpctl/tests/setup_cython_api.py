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

import os.path

import setuptools
from Cython.Build import build_ext

import dpctl


def get_includes():
    # path to dpctl/include
    dpctl_incl_dir = dpctl.get_include()
    # path to folder where __init__.pxd resides
    dpctl_pxd_dir = os.path.dirname(os.path.dirname(dpctl_incl_dir))
    return [dpctl_incl_dir, dpctl_pxd_dir]


ext = setuptools.Extension(
    "_cython_api",
    ["_cython_api.pyx"],
    include_dirs=get_includes(),
    language="c++",
)

setuptools.setup(
    name="test_cython_api",
    version="0.0.0",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
)
