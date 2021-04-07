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

import sys
from os.path import join, exists, abspath, dirname
from os import getcwd
from os import environ
from Cython.Build import cythonize


def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info
    import numpy as np
    import dpctl

    config = Configuration("", parent_package, top_path)

    oneapi_root = environ.get("ONEAPI_ROOT", None)
    if not oneapi_root:
        raise ValueError("ONEAPI_ROOT must be set, typical value is /opt/intel/oneapi")

    mkl_info = {
        "include_dirs": [join(oneapi_root, "mkl", "include")],
        "library_dirs": [
            join(oneapi_root, "mkl", "lib"),
            join(oneapi_root, "mkl", "lib", "intel64"),
        ],
        "libraries": [
            "mkl_sycl",
            "mkl_intel_ilp64",
            "mkl_tbb_thread",
            "mkl_core",
            "tbb",
            "iomp5",
        ],
    }

    mkl_include_dirs = mkl_info.get("include_dirs")
    mkl_library_dirs = mkl_info.get("library_dirs")
    mkl_libraries = mkl_info.get("libraries")

    pdir = dirname(__file__)
    wdir = join(pdir)

    eca = ["-Wall", "-Wextra", "-fsycl", "-fsycl-unnamed-lambda"]

    config.add_extension(
        name="syclbuffer",
        sources=[
            join(pdir, "_buffer_example.pyx"),
            join(wdir, "use_sycl_buffer.cpp"),
            join(wdir, "use_sycl_buffer.h"),
        ],
        include_dirs=[wdir, np.get_include(), dpctl.get_include()] + mkl_include_dirs,
        libraries=["sycl"] + mkl_libraries,
        runtime_library_dirs=mkl_library_dirs,
        extra_compile_args=eca,  # + ['-O0', '-g', '-ggdb'],
        extra_link_args=["-fPIC"],
        language="c++",
    )

    config.ext_modules = cythonize(config.ext_modules, include_path=[pdir, wdir])
    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(configuration=configuration)
