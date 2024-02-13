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

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

import dpctl


class custom_build_ext(build_ext):
    def build_extensions(self):
        self.compiler.set_executable("compiler_so", "icpx -fsycl -fPIC")
        self.compiler.set_executable("compiler_cxx", "icpx -fsycl -fPIC")
        self.compiler.set_executable(
            "linker_so",
            "icpx -fsycl -shared -fpic -fsycl-device-code-split=per_kernel",
        )
        build_ext.build_extensions(self)


ext_modules = [
    Extension(
        name="syclbuffer._syclbuffer",
        sources=[
            "syclbuffer/_buffer_example.pyx",
        ],
        depends=[
            "src/use_sycl_buffer.hpp",
        ],
        include_dirs=[
            ".",
            "./src",
            dpctl.get_include(),
        ],
        extra_compile_args=[
            "-Wall",
            "-Wextra",
            "-fsycl",
        ],
        extra_link_args=["-fPIC"],
        language="c++",
    )
]

setup(
    name="syclbuffer",
    version="0.0.0",
    description="An example of Cython extension calling SYCL routines",
    long_description="""
    Example of using SYCL to work on host allocated NumPy array using
    SYCL buffers and SYCL functions.

    See README.md for more details.
    """,
    license="Apache 2.0",
    author="Intel Corporation",
    url="https://github.com/IntelPython/dpctl",
    ext_modules=ext_modules,
    cmdclass={"build_ext": custom_build_ext},
)
