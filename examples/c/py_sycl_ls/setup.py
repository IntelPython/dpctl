#                      Data Parallel Control (dpctl)
#
# Copyright 2022 Intel Corporation
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
import sysconfig

from setuptools import Extension, setup

import dpctl

setup(
    name="py_sycl_ls",
    version="0.0.1",
    description="An example of C extension calling SYCLInterface routines",
    long_description="""
    Example of using SYCLInterface.

    See README.md for more details.
    """,
    license="Apache 2.0",
    author="Intel Corporation",
    url="https://github.com/IntelPython/dpctl",
    ext_modules=[
        Extension(
            name="py_sycl_ls._py_sycl_ls",
            sources=[
                "src/py_sycl-ls.c",
            ],
            include_dirs=[
                dpctl.get_include(),
                os.path.join(sysconfig.get_paths()["include"], ".."),
            ],
            library_dirs=[
                os.path.join(dpctl.get_include(), ".."),
            ],
            libraries=["DPCTLSyclInterface"],
            runtime_library_dirs=[
                os.path.join(dpctl.get_include(), ".."),
            ],
            extra_compile_args=[
                "-Wall",
                "-Wextra",
            ],
            extra_link_args=["-fPIC"],
            language="c",
        )
    ],
)
