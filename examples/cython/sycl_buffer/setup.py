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

import numpy as np
from setuptools import Extension, setup

import dpctl

setup(
    name="syclbuffer",
    version="0.0.0",
    description="An example of Cython extension calling SYCL routines",
    long_description="""
    Example of using SYCL to work on host allocated NumPy array using
    SYCL buffers by calling oneMKL functions.

    See README.md for more details.
    """,
    license="Apache 2.0",
    author="Intel Corporation",
    url="https://github.com/IntelPython/dpctl",
    ext_modules=[
        Extension(
            name="syclbuffer",
            sources=[
                "_buffer_example.pyx",
                "use_sycl_buffer.cpp",
            ],
            include_dirs=[".", np.get_include(), dpctl.get_include()],
            libraries=["sycl"]
            + [
                "mkl_sycl",
                "mkl_intel_ilp64",
                "mkl_tbb_thread",
                "mkl_core",
                "tbb",
                "iomp5",
            ],
            runtime_library_dirs=[],
            extra_compile_args=[
                "-Wall",
                "-Wextra",
                "-fsycl",
                "-fsycl-unnamed-lambda",
            ],
            extra_link_args=["-fPIC"],
            language="c++",
        )
    ],
)
