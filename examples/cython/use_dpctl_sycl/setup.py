#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2022 Intel Corporation
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

import Cython.Build
import setuptools
import setuptools.command.build_ext

import dpctl


class custom_build_ext(
    setuptools.command.build_ext.build_ext, Cython.Build.build_ext
):
    def build_extensions(self):
        self.compiler.set_executable("compiler_so", "icx -fsycl -fPIC")
        self.compiler.set_executable("compiler_cxx", "icpx -fsycl -fPIC")
        self.compiler.set_executable(
            "linker_so",
            "icpx -fsycl -shared -fpic -fsycl-device-code-split=per_kernel",
        )
        super().build_extensions()


ext = setuptools.Extension(
    "use_dpctl_sycl._cython_api",
    ["./use_dpctl_sycl/_cython_api.pyx"],
    include_dirs=[dpctl.get_include(), "./use_dpctl_sycl"],
    language="c++",
)

setuptools.setup(
    name="use_dpctl_sycl",
    version="0.0.0",
    ext_modules=[ext],
    cmdclass={"build_ext": custom_build_ext},
)
